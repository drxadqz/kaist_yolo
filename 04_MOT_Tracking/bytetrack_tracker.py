import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(box1, box2):
    """
    计算两个检测框的IoU，和SORT、融合代码逻辑完全一致，保证对比公平性
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
    """
    单个目标的卡尔曼滤波跟踪器，和SORT完全一致的状态定义，保证对比实验的唯一变量是匹配逻辑
    状态向量：[x,y,s,r,dx,dy,ds,dr]
    x,y: 目标框中心坐标；s: 目标面积；r: 宽高比；dx,dy,ds,dr: 对应速度分量
    """
    count = 0  # 全局轨迹ID计数器

    def __init__(self, bbox):
        """
        初始化跟踪器
        :param bbox: 检测框格式 (x1,y1,x2,y2,conf)，和SORT完全兼容
        """
        # 卡尔曼滤波器定义，和SORT完全一致
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # 状态转移矩阵F：匀速运动模型
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        # 观测矩阵H：仅观测位置、面积、宽高比，无法观测速度
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        # 噪声参数，和SORT完全一致，保证对比公平性
        self.kf.R[2:, 2:] *= 10.  # 观测噪声
        self.kf.P[4:, 4:] *= 1000.  # 初始速度协方差，高不确定性
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01  # 过程噪声
        self.kf.Q[4:, 4:] *= 0.01

        # 用初始检测框初始化卡尔曼状态
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        self.kf.x[:4] = np.array([(x1 + x2) / 2, (y1 + y2) / 2, w * h, w / h]).reshape((4, 1))

        # 跟踪器状态变量，和SORT完全兼容
        self.time_since_update = 0  # 距离上次更新的帧数
        self.id = KalmanBoxTracker.count  # 唯一轨迹ID
        KalmanBoxTracker.count += 1
        self.history = []  # 轨迹历史
        self.hits = 0  # 总命中次数
        self.hit_streak = 0  # 连续命中帧数
        self.age = 0  # 跟踪器存在帧数
        self.conf = bbox[4]  # 检测框置信度

    def update(self, bbox):
        """用匹配的检测框更新卡尔曼滤波器状态，和SORT完全一致"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # 转换检测框为观测格式
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        self.kf.update(np.array([(x1 + x2) / 2, (y1 + y2) / 2, w * h, w / h]).reshape((4, 1)))
        self.conf = bbox[4]

    def predict(self):
        """预测当前帧目标状态，返回预测框，和SORT完全一致"""
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
        """获取当前跟踪器的检测框，格式(x1,y1,x2,y2)，和SORT完全一致"""
        x = self.kf.x[0]
        y = self.kf.x[1]
        s = self.kf.x[2]
        r = self.kf.x[3]
        w = np.sqrt(s * r)
        h = s / w
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape((4,))


class BYTETracker:
    """
    ByteTrack跟踪器主类，接口和SORT类100%兼容，直接替换即可使用
    核心改进：高低置信度检测框分两次匹配，提升遮挡目标跟踪性能
    """

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3,
                 high_conf_thresh=0.6, low_conf_thresh=0.1):
        """
        初始化ByteTrack跟踪器，默认参数和SORT对齐，适配KAIST行人场景
        :param max_age: 目标最大消失帧数，和SORT默认一致
        :param min_hits: 最小连续命中帧数，和SORT默认一致
        :param iou_threshold: 匹配IoU阈值，和SORT默认一致
        :param high_conf_thresh: 高置信度阈值，第一次匹配使用
        :param low_conf_thresh: 低置信度阈值，第二次匹配使用
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.trackers = []  # 当前活跃的跟踪器
        self.frame_count = 0  # 已处理帧数

    def update(self, dets=np.empty((0, 5))):
        """
        每一帧更新跟踪器，核心调用接口，和SORT完全兼容
        :param dets: 检测框，格式[N,5]，每个元素为(x1,y1,x2,y2,conf)，直接复用融合输出
        :return: 跟踪结果，格式[N,6]，每个元素为(x1,y1,x2,y2,track_id,conf)，和SORT完全一致
        """
        self.frame_count += 1

        # ===================== ByteTrack核心步骤1：检测框置信度分级 =====================
        # 高置信度检测框：conf >= high_conf_thresh
        high_conf_dets = dets[dets[:, 4] >= self.high_conf_thresh]
        # 低置信度检测框：low_conf_thresh <= conf < high_conf_thresh
        low_conf_dets = dets[(dets[:, 4] >= self.low_conf_thresh) & (dets[:, 4] < self.high_conf_thresh)]

        # ===================== 步骤2：所有轨迹预测当前帧位置 =====================
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

        # ===================== ByteTrack核心步骤2：第一次匹配（高置信度检测框+轨迹）=====================
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            high_conf_dets, trks, self.iou_threshold
        )
        # 用匹配的高置信度检测框更新轨迹
        for m in matched:
            self.trackers[m[1]].update(high_conf_dets[m[0], :])

        # ===================== ByteTrack核心步骤3：第二次匹配（低置信度检测框+未匹配轨迹）=====================
        # 仅对第一次匹配中未匹配上的轨迹，和低置信度检测框做匹配
        if len(unmatched_trks) > 0 and len(low_conf_dets) > 0:
            # 提取未匹配的轨迹
            unmatched_trk_boxes = trks[unmatched_trks]
            # 第二次匹配，低置信度匹配阈值放宽
            low_matched, low_unmatched_dets, low_unmatched_trks = self.associate_detections_to_trackers(
                low_conf_dets, unmatched_trk_boxes, self.iou_threshold * 0.5
            )
            # 用匹配的低置信度检测框更新轨迹
            for m in low_matched:
                trk_idx = unmatched_trks[m[1]]
                self.trackers[trk_idx].update(low_conf_dets[m[0], :])
            # 更新未匹配的轨迹和检测框
            unmatched_trks = [unmatched_trks[i] for i in low_unmatched_trks]
            unmatched_dets = np.concatenate([unmatched_dets, low_unmatched_dets]) if len(
                low_unmatched_dets) > 0 else unmatched_dets

        # ===================== 步骤4：为未匹配的高置信度检测框创建新轨迹 =====================
        for i in unmatched_dets:
            if i < len(high_conf_dets):
                target_box = high_conf_dets[i, :]
            else:
                target_box = low_conf_dets[i - len(high_conf_dets), :]
            trk = KalmanBoxTracker(target_box)
            self.trackers.append(trk)

        # ===================== 步骤5：清理过期轨迹，输出有效结果（和SORT完全一致）=====================
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # 仅输出满足连续命中要求的有效轨迹，过滤新生误检
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.conf])).reshape(1, -1))
            i -= 1
            # 删除超过最大消失帧数的过期轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        """匈牙利算法实现检测框与轨迹的匹配，和SORT逻辑一致，可自定义阈值"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

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
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # 过滤掉IoU低于阈值的无效匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# 单文件测试代码，直接运行可验证跟踪器是否正常
if __name__ == "__main__":
    # 统一numpy数组打印格式，关闭科学计数法，固定2位小数，彻底解决两帧输出格式不一致问题
    np.set_printoptions(suppress=True, precision=2, floatmode='fixed', linewidth=100)

    tracker = BYTETracker()
    # 模拟两帧检测框，验证跟踪器ID分配逻辑
    # 第一帧
    dets_frame1 = np.array([[100, 100, 200, 300, 0.9], [300, 200, 400, 400, 0.8]])
    tracks_frame1 = tracker.update(dets_frame1)
    print("=" * 50)
    print("第一帧跟踪结果：")
    print(f"跟踪框格式：x1,y1,x2,y2,轨迹ID,置信度")
    print(tracks_frame1)
    print("=" * 50)

    # 第二帧（模拟目标移动+低置信度遮挡目标）
    dets_frame2 = np.array([[105, 102, 205, 302, 0.92], [303, 201, 403, 401, 0.25]])
    tracks_frame2 = tracker.update(dets_frame2)
    print("第二帧跟踪结果（ID应与第一帧一致，低置信度目标正常跟踪）：")
    print(tracks_frame2)
    print("=" * 50)