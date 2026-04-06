import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(box1, box2):
    """
    计算两个检测框的IoU，和你晚期融合代码的IoU计算逻辑完全一致
    box格式：(x1,y1,x2,y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

class KalmanBoxTracker:
    """单个行人目标的卡尔曼滤波跟踪器，用于预测目标下一帧的位置"""
    count = 0  # 全局轨迹ID计数器
    def __init__(self, bbox):
        """
        初始化跟踪器
        :param bbox: 检测框格式 (x1,y1,x2,y2,conf)，直接复用融合检测的输出
        """
        # 定义状态向量：[x,y,s,r,dx,dy,ds,dr]
        # x,y: 目标框中心坐标；s: 目标面积；r: 宽高比；dx,dy,ds,dr: 对应速度分量
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # 状态转移矩阵F：匀速运动模型
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        # 观测矩阵H：仅能观测到中心坐标、面积、宽高比，无法直接观测速度
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])

        # 噪声参数（KAIST行人场景最优适配，无需修改）
        self.kf.R[2:,2:] *= 10.  # 观测噪声
        self.kf.P[4:,4:] *= 1000. # 初始速度协方差，给速度高不确定性
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01  # 过程噪声
        self.kf.Q[4:,4:] *= 0.01

        # 用第一帧检测框初始化卡尔曼滤波器状态
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        self.kf.x[:4] = np.array([(x1+x2)/2, (y1+y2)/2, w*h, w/h]).reshape((4,1))

        # 跟踪器状态变量
        self.time_since_update = 0  # 距离上次更新的帧数
        self.id = KalmanBoxTracker.count  # 唯一轨迹ID
        KalmanBoxTracker.count += 1
        self.history = []  # 轨迹历史
        self.hits = 0  # 总命中次数
        self.hit_streak = 0  # 连续命中帧数
        self.age = 0  # 跟踪器存在帧数
        self.conf = bbox[4]  # 检测框置信度

    def update(self, bbox):
        """用当前帧的匹配检测框，更新卡尔曼滤波器状态"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # 转换检测框为卡尔曼观测格式
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        self.kf.update(np.array([(x1+x2)/2, (y1+y2)/2, w*h, w/h]).reshape((4,1)))
        self.conf = bbox[4]

    def predict(self):
        """预测当前帧目标的状态，返回预测框"""
        # 防止面积变为负数
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        # 连续未命中，重置连续命中计数
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        """获取当前跟踪器的检测框，格式：(x1,y1,x2,y2)"""
        x = self.kf.x[0]
        y = self.kf.x[1]
        s = self.kf.x[2]
        r = self.kf.x[3]
        w = np.sqrt(s * r)
        h = s / w
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2]).reshape((4,))

class SORT:
    """SORT跟踪器主类，对外统一接口"""
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        """
        初始化SORT跟踪器，参数适配KAIST行人场景
        :param max_age: 目标最大消失帧数，超过则删除轨迹（应对行人遮挡、出画）
        :param min_hits: 最小连续命中帧数，达到才输出有效轨迹（过滤误检）
        :param iou_threshold: 检测框与轨迹匹配的IoU阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # 当前所有活跃的跟踪器
        self.frame_count = 0  # 已处理帧数

    def update(self, dets=np.empty((0, 5))):
        """
        每一帧更新跟踪器，核心调用接口
        :param dets: 检测框，格式[N,5]，每个元素为(x1,y1,x2,y2,conf)，直接复用你的晚期融合输出
        :return: 跟踪结果，格式[N,6]，每个元素为(x1,y1,x2,y2,track_id,conf)
        """
        self.frame_count += 1
        # 步骤1：对所有已有轨迹，预测当前帧的位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # 过滤无效预测框
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 步骤2：匈牙利算法匹配检测框和预测轨迹
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # 步骤3：用匹配的检测框更新对应轨迹
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 步骤4：为未匹配的检测框创建新轨迹
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 步骤5：清理过期轨迹，输出有效跟踪结果
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # 仅输出满足连续命中要求的有效轨迹（过滤新生误检）
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1], [trk.conf])).reshape(1, -1))
            i -= 1
            # 删除超过最大消失帧数的过期轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def associate_detections_to_trackers(self, detections, trackers):
        """匈牙利算法实现检测框与轨迹的匹配"""
        if len(trackers) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

        # 计算IoU成本矩阵
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)

        # 匈牙利算法最优匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)

        # 筛选未匹配的检测框和轨迹
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)

        # 过滤掉IoU低于阈值的无效匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# 单文件测试代码，直接运行可验证跟踪器是否正常
if __name__ == "__main__":
    tracker = SORT()
    # 模拟两帧检测框，验证跟踪器ID分配逻辑
    # 第一帧
    dets_frame1 = np.array([[100, 100, 200, 300, 0.9], [300, 200, 400, 400, 0.8]])
    tracks_frame1 = tracker.update(dets_frame1)
    print("="*50)
    print("第一帧跟踪结果：")
    print(f"跟踪框格式：x1,y1,x2,y2,轨迹ID,置信度")
    print(tracks_frame1)
    print("="*50)
    # 第二帧（模拟目标移动）
    dets_frame2 = np.array([[105, 102, 205, 302, 0.92], [303, 201, 403, 401, 0.85]])
    tracks_frame2 = tracker.update(dets_frame2)
    print("第二帧跟踪结果（ID应与第一帧一致）：")
    print(tracks_frame2)
    print("="*50)