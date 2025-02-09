import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from lib.models.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from utils.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os.path as osp


# from apex import amp


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)
        self.log_dir = osp.join(opt.save_dir, f"logs_{time.strftime('%Y-%m-%d-%H-%M')}")
        self.writer = SummaryWriter(self.log_dir)
        # breakpoint()
        # self.writer.add_graph(self.model_with_loss.model, torch.zeros(opt.batch_size, opt.input_h, opt.input_w, 3))
        self.val_writer = SummaryWriter(osp.join(self.log_dir, 'val'))
        self.global_steps = 0

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        raise NotImplementedError

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)

            results.setdefault('img_id', []).extend(batch['meta']['img_id'])
            boxes = ctdet_decode(output['hm'], output['wh'], output['reg'], K=self.opt.K)
            meta = batch['meta']
            c = meta['c'].numpy()
            s = meta['s'].numpy()
            _, _, out_h, out_w = output['hm'].shape
            dets = ctdet_post_process(boxes.cpu().numpy(), [c], [s], out_h, out_w, opt.num_classes)
            # breakpoint()
            results.setdefault('detections', []).extend(dets)
            loss = loss.mean()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase='val',
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                # breakpoint()
                loss_t = loss_stats[l] or torch.zeros(1)
                avg_loss_stats[l].update(loss_t.mean().item(), batch['input'].size(0))
                Bar.suffix += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                self.val_writer.add_scalar(l, avg_loss_stats[l].avg, self.global_steps, 1)

            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats
            # break
        ret = {k: v.avg for k, v in avg_loss_stats.items()}

        for l in avg_loss_stats:
            self.val_writer.add_scalar(l, avg_loss_stats[l].avg, self.global_steps)
        bar.next()
        bar.finish()
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def train(self, epoch, data_loader):
        model_with_loss = self.model_with_loss
        model_with_loss.train()
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            boxes = batch['reg_mask'].sum()
            self.writer.add_scalar('num_boxes', boxes.item(), self.global_steps, 10)
            if boxes == 0: continue
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            self.optimizer.zero_grad()
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #   scaled_loss.backward()
            loss.backward()
            self.optimizer.step()
            self.global_steps += 1
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase='train',
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                self.writer.add_scalar(l, avg_loss_stats[l].avg, self.global_steps, 10) # wait for 10 seconds

            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats
            # break
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        bar.finish()
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results
