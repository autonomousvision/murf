import torch
import os
import numpy as np
import telepot
from collections import OrderedDict


@torch.no_grad()
def summarize_metrics(metrics, out_dir, it=None, ep=None):
    head_info = ""
    if it is not None:
        head_info = f" at Iteration [{it}]"
    if ep is not None:
        head_info = f" at Epoch [{ep}]"

    dataset_metrics = {}
    for dataname, raw_metrics in metrics.items():
        dataset_metrics[dataname] = {}
        header = f"------------ {dataname.upper()} Nearest 3{head_info} ------------"
        all_msgs = [header]
        cur_scene = ""
        for view_id, view_metrics in raw_metrics.items():
            if view_id.split('_')[0] != cur_scene:
                if cur_scene != "":  # summarise scene buffer and log
                    scene_info = f"====> scene: {cur_scene},"
                    for k, v in scene_metrics.items():
                        scene_info = scene_info + f" {k}: {float(np.array(v).mean())},"
                    all_msgs.append(scene_info)
                else:  # init dataset
                    dataset_metrics[dataname] = OrderedDict({k:[] for k in view_metrics.keys()})
                # reset scene buffer
                cur_scene = view_id.split('_')[0]
                scene_metrics = {k:[] for k in view_metrics.keys()}
            # log view
            view_info = f"==> view: {view_id},"
            for k, v in view_metrics.items():
                view_info = view_info + f" {k}: {float(v)},"
                scene_metrics[k].append(v)
                dataset_metrics[dataname][k].append(v)
            all_msgs.append(view_info)
        # summarise dataset
        data_info = f"======> {dataname.upper()}{head_info},"
        for k, v in dataset_metrics[dataname].items():
            data_info = data_info + f" {k}: {float(np.array(v).mean())},"
        all_msgs.append(data_info)
        with open(os.path.join(out_dir, f"0results_{dataname}.txt"), "a+") as f:
            f.write('\n'.join(all_msgs))
            f.write('\n')

        # a single file only log mean metrics for simplicity
        with open(os.path.join(out_dir, f"0results_{dataname}_mean.txt"), "a+") as f:
            f.write(data_info)
            f.write('\n')

    return dataset_metrics


@torch.no_grad()
def summarize_metrics_list(metrics, out_dir, it=None, ep=None):
    # metrics: dict of dataname: list
    head_info = ""
    if it is not None:
        head_info = f"Iteration [{it}]"
    if ep is not None:
        head_info = f"Epoch [{ep}]"

    dataset_metrics = {}
    for dataname, raw_metrics in metrics.items():
        dataset_metrics[dataname] = {}

        all_msgs = [head_info]

        all_metrics = OrderedDict({k:[] for k in raw_metrics[0].keys()})
        for i, single_metric in enumerate(raw_metrics):
            for k, v in single_metric.items():
                all_metrics[k].append(v)

        data_info = ""
        for k, v in all_metrics.items():
            dataset_metrics[dataname][k] = float(np.array(v).mean())
            data_info = data_info + f"{k}: {dataset_metrics[dataname][k]}, "

        all_msgs.append(data_info)

        with open(os.path.join(out_dir, f"0results_{dataname}_mean.txt"), "a+") as f:
            f.write(data_info)
            f.write('\n')

    return dataset_metrics


def summarize_loss(loss, loss_weight):
    loss_all = 0.
    assert("all" not in loss)
    # weigh losses
    for key in loss:
        assert(key in loss_weight)
        assert(loss[key].shape==())
        if loss_weight[key] is not None:
            # skip nan loss
            if torch.isinf(loss[key]):
                print("loss {} is Inf".format(key))
                continue
            
            if torch.isnan(loss[key]):
                print("loss {} is NaN".format(key))
                continue

            # assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
            # assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
            loss_all = loss_all + float(loss_weight[key]) * loss[key]
    loss.update(all=loss_all)
    return loss


class TGDebugMessager(object):
    """Tools to send and update logs to the telegram bot."""
    def __init__(self, tg_token, tg_chat_id):
        super(TGDebugMessager, self).__init__()
        self.tg_bot = telepot.Bot(token=tg_token)
        self.tg_chat_id = tg_chat_id
        self.reset_msg()

    def send_msg(self, msg, parse_mode):
        full_msg = self.update_full_msg(msg)
        sent_msg = self.tg_bot.sendMessage(chat_id=self.tg_chat_id, text=full_msg, parse_mode=parse_mode)
        self.msg_id = telepot.message_identifier(sent_msg)
    
    def reset_msg(self):
        self.msg_id = None
        self.msg_text = []

    def update_full_msg(self, msg):
        self.msg_text.append(msg)
        out_text = '\n'.join(self.msg_text)
        return out_text

    def edit_msg(self, msg, parse_mode):
        assert self.msg_id is not None, "Cannot find the original message."

        full_msg = self.update_full_msg(msg)
        self.tg_bot.editMessageText(self.msg_id, full_msg, parse_mode=parse_mode)

    def __call__(self, msg, parse_mode="HTML", **kwds):
        try:
            if self.msg_id is None:
                self.send_msg(msg, parse_mode)
            else:
                self.edit_msg(msg, parse_mode)
        except:
            print("[WARNING] Telegram bot fails to send message, continue.")
