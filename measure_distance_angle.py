"""
单目视觉目标物测量系统
基于A4纸边框的视觉测量系统

题目要求：
- 距离D范围：80cm~150cm
- 基本目标物：圆形、等边三角形、正方形（边长10-16cm）
- 发挥目标物：多个正方形（边长6-12cm），可能重叠
- A4纸规格：210mm x 297mm，四边2cm黑色边框
- 测量误差：D≤8cm，x≤1.5cm，角度≤10°

硬件：
- 树莓派5 2GB
- USB摄像头 640x480
- SPI LCD显示屏 (ILI9341V 2.8寸 240x320)
- GPIO按键 x2
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any  # 类型标注：给代码加 “标签”，说明变量是什么类型
from dataclasses import dataclass, field  # 数据类：用来打包一组数据
from enum import Enum  # 枚举：定义固定选项
import time


# from 包名.模块名 import 类名
from package.SPI import SPILCD
from package.GPIO import GPIOButtons

# 尝试导入树莓派GPIO和SPI库
GPIO = None
Image = None
ImageDraw = None
ImageFont = None
spidev = None

try:
    import RPi.GPIO as GPIO
    from PIL import Image, ImageDraw, ImageFont
    import spidev

    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False
    print("警告: 未检测到树莓派GPIO库，使用模拟模式")


class ShapeType(Enum):  # 枚举类（不可更改）
    """几何图形类型"""
    UNKNOWN = "unknown"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"


@dataclass  # 变量名: 类型, 数据类自带魔术方法
class GeometryInfo:
    """几何图形信息"""
    shape_type: ShapeType = ShapeType.UNKNOWN   # 形状类型
    size: float = 0.0   # 真实尺寸（毫米）
    pixel_size: float = 0.0  # 边长像素大小
    rotation_angle: float = 0.0
    # field (default_factory = 类名) = 给数据类一个「自动创建空对象」的默认值,数据类的特殊写法
    center: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    contour: Optional[np.ndarray] = None    # optional:可选（数据类型为np.ndarray/None）
    valid: bool = False


@dataclass
class MeasurementResult:
    """测量结果"""
    distance: float = 0.0
    # field (default_factory = 类名) = 给数据类一个「自动创建空对象」的默认值
    geometry: GeometryInfo = field(default_factory=GeometryInfo)
    valid: bool = False  # 是否测量成功
    min_size: float = 0.0   # 进阶模式最小正方形尺寸 / 角度
    min_angle: float = 0.0


class A4BorderDetector:
    """A4纸边框检测器"""

    A4_WIDTH, A4_HEIGHT, BORDER_WIDTH = 210.0, 297.0, 20.0

    def detect_a4_border(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测A4纸边框"""
        """
        image: np.ndarray输入的图像
        Optional[Dict[str, Any]]: {'corners': corners,'pixel_width': pixel_width,'pixel_height': pixel_height} 或者 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应二值化
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        # cv2.RETR_EXTERNAL：只找最外层，就是外框
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # float ('inf') = 无穷大, infinity = 无穷大
        best_quad, best_score = None, float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if not 5000 < area < 200000:
                continue

            perimeter = cv2.arcLength(contour, True)
            # 精度（epsilon）：误差不超过周长的 2%
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                # ( (cx, cy), (w, h), angle )
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width == 0 or height == 0:
                    continue

                aspect_ratio = max(width, height) / min(width, height)
                expected_ratio = self.A4_HEIGHT / self.A4_WIDTH
                ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio

                if ratio_error < 0.3 and ratio_error < best_score:
                    best_score = ratio_error
                    best_quad = {'contour': contour, 'approx': approx, 'rect': rect}

        if best_quad is None:
            return None

        corners = self._order_corners(best_quad['approx'])
        # np.linalg.norm算距离
        pixel_width = np.linalg.norm(corners[1] - corners[0])
        pixel_height = np.linalg.norm(corners[2] - corners[1])

        return {
            'corners': corners,
            'pixel_width': pixel_width,
            'pixel_height': pixel_height
        }

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """角点排序：左上、右上、右下、左下"""
        """
        将角点排序成左上、右上、右下、左下
        corner: 角点(n, 1, 2)
        np.ndarray: 排序后的角点
        """
        corners = corners.reshape(4, 2)
        # np.argsort:排序从小到大并相应返回索引
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


class GeometryDetector:
    """几何图形检测器"""

    BASIC_SIZE_RANGE = (100.0, 160.0)
    ADVANCED_SIZE_RANGE = (60.0, 120.0)

    # 主函数
    def detect_all_geometries(self, warped_image: np.ndarray,
                              is_advanced: bool = False) -> List[GeometryInfo]:
        """检测所有几何图形"""
        """
        warped_image: 经矫正后的图形
        is_advanced: 是否是进阶模式
        List[GeometryInfo]: 存放GeometryInfo类对象的列表
        """
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # 20：黑框宽度mm，210：白纸宽
        border_px = int(w * (20.0 / 210.0))
        # 裁剪黑框
        """
        [起始 : 结束]      # 左闭右开，包含起始，不包含结束
        [正数]            # 从前往后数
        [负数]            # 从后往前数
        """
        roi = gray[border_px:-border_px, border_px:-border_px]

        # 基础检测
        geometries = self._detect_by_contours(roi, border_px)

        if is_advanced:
            """
            append：把一整个东西直接放进列表
            extend：把里面的元素拆出来，一个个放进列表
            """
            geometries.extend(self._detect_by_watershed(roi, border_px))

        return self._remove_duplicates(geometries)

    #  普通检测
    def _detect_by_contours(self, roi: np.ndarray, border_px: int) -> List[GeometryInfo]:
        """轮廓检测"""
        """
        roi: 感兴趣区域
        border_px: 边框像素大小，用于还原图像
        List[GeometryInfo]: 存放每个元素是GeometryInfo对象的列表
        """
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        geometries: List[GeometryInfo] = []

        if hierarchy is not None:
            for contour in contours:
                # 分类几何图形
                geo = self._classify_geometry(contour)
                if geo.valid:
                    geo.center = (geo.center[0] + border_px, geo.center[1] + border_px)
                    geometries.append(geo)

        return geometries

    # 高级检测
    def _detect_by_watershed(self, roi: np.ndarray, border_px: int) -> List[GeometryInfo]:
        """分水岭算法检测重叠图形"""
        """
        roi: 感兴趣区域
        border_px: 边框像素大小，用于还原图像
        List[GeometryInfo]: 存放GeometryInfo类对象的列表
        """
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 距离变换（找图形的中心）, cv2.DIST_L2:距离公式
        # 计算每个像素离最近黑色边框的距离
        """
        靠近边框 → 距离小 → 暗
        图形正中心 → 距离最大 → 最亮
        """
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # 归一化函数:  把数值缩放到 0~255 之间
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 确定 “肯定是图形” 的区域, 只保留最亮的中心点
        _, sure_fg = cv2.threshold(dist_transform, int(0.3 * dist_transform.max()), 255, 0)
        sure_fg = np.uint8(sure_fg)

        sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
        # cv2.subtract：图像做减法， 去除背景
        # unknown：不确定区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 物体数量， 一张和原图一样大的标签图，每个物体被标上不同数字（1、2、3...），背景算0
        # markers: np.ndarray (dtype=np.int32)
        """
        原图：
        黑 黑 黑 黑 黑
        黑 白 白 黑 黑
        黑 白 白 黑 白
        黑 黑 黑 黑 白
        markers：
        0  0  0  0  0
        0  1  1  0  0
        0  1  1  0  2
        0  0  0  0  2
        """
        num_markers, markers = cv2.connectedComponents(sure_fg)
        if num_markers < 2:
            return []

        markers = markers + 1           # 所有标签+1，背景变成1
        markers[unknown == 255] = 0     # 未知区域设为0
        # cv2.watershed( 图像 , 标记图 )
        """
        原图：
        0 0 0 0 0
        0 1 0 2 0
        0 0 0 0 0
        markers：
        1  -1  -1  -1  1
        1   2  -1   3  1
        1  -1  -1  -1  1
        1：背景
        1和-1都是边界线
        边界线是只在未知区域的基础上画出
        """
        markers = cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)
        """
        - 外部边界 ：由原始二值化图像（binary）确定
        - 内部边界 ：由分水岭算法在重叠处自动创建
        - 完整边界 ：两者的组合
        """
        # 定义存放GeometryInfo类对象的列表
        geometries: List[GeometryInfo] = []
        for marker_id in range(2, min(num_markers + 1, 15)):
            # 只保留当前编号的图形，其他全部变黑
            # 布尔判断， 等于这个编号 → True， 不等于 → False， 再转成0和
            mask = np.uint8(markers == marker_id)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = max(contours, key=cv2.contourArea)
                if 500 < cv2.contourArea(contour) < 80000:
                    geo = self._classify_geometry(contour)
                    if geo.valid:
                        geo.center = (geo.center[0] + border_px, geo.center[1] + border_px)
                        geometries.append(geo)

        return geometries

    def _remove_duplicates(self, geometries: List[GeometryInfo]) -> List[GeometryInfo]:
        """去重"""
        """
        geometries: 存放GeometryInfo类对象的列表
        List[GeometryInfo]: 返回存放GeometryInfo类对象的列表
        """
        # 创建空列表，存放 “不重复” 的图形
        unique: List[GeometryInfo] = []
        # geo是一个列表， 每个元素是一个图形对象
        for geo in geometries:
            # 第一次循环， 不进入if
            if not any(np.hypot(geo.center[0] - u.center[0],
                                geo.center[1] - u.center[1]) < 15 for u in unique):
                unique.append(geo)
        return unique

    def _classify_geometry(self, contour: np.ndarray) -> GeometryInfo:
        """图形分类"""
        """
        contour: 图像的轮廓
        GeometryInfo: GeometryInfo的类对象存放图形信息
        """
        # 创建类对象
        geo = GeometryInfo(contour=contour)

        # M：返回的结果（字典，里面全是特征值）
        """
        M['m00']→ 轮廓的面积
        M['m10']→ 横向总和
        M['m01']→ 纵向总和
        3个点
        (2, 3)
        (4, 5)
        (6, 7)
        计算：
        m00 = 1+1+1 = 3（总点数）
        m10 = 2+4+6 = 12（x 总和）
        m01 = 3+5+7 = 15（y 总和）
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            geo.center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        num_corners = len(approx)

        if num_corners == 3:
            geo = self._measure_polygon(geo, approx, ShapeType.TRIANGLE, 3, 120, 0.03)
        elif num_corners == 4:
            geo = self._measure_polygon(geo, approx, ShapeType.SQUARE, 4, 90, 0.008)
        elif num_corners > 6:
            geo = self._measure_circle(geo, contour)

        return geo

    # polygon: 多边形
    def _measure_polygon(self, geo: GeometryInfo, approx: np.ndarray,
                         shape_type: ShapeType, n_sides: int,
                         angle_mod: int, variance_threshold: float) -> GeometryInfo:
        """测量多边形（三角形/正方形）"""
        """
        geo: GeometryInfo类对象
        approx: 角点
        shape_type: 形状种类
        n_sides: 边数
        angle_mod: 角度模式
        variance_threshold: 方差阈值
        GeometryInfo: 图像信息
        """
        side_lengths = []
        for i in range(n_sides):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % n_sides][0]
            side_lengths.append(float(np.hypot(p1[0] - p2[0], p1[1] - p2[1])))

        mean_side = float(np.mean(side_lengths))
        # np.var：求方差， 两数相除 = 归一化方差
        variance = float(np.var(side_lengths) / (mean_side ** 2))

        if variance < variance_threshold:
            geo.shape_type = shape_type
            geo.pixel_size = mean_side
            geo.valid = True

            p1, p2 = approx[0][0], approx[1][0]
            angle = math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))
            geo.rotation_angle = angle % angle_mod

        return geo

    def _measure_circle(self, geo: GeometryInfo, contour: np.ndarray) -> GeometryInfo:
        """测量圆形"""
        """
        geo: GeometryInfo类对象
        contour: 图像轮廓
        GeometryInfo: 返回一个GeometryInfo类对象
        """
        (_, _), radius = cv2.minEnclosingCircle(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 圆形度 = 4π * 面积 / (周长²)
        if perimeter > 0 and 4 * np.pi * area / (perimeter ** 2) > 0.85:
            geo.shape_type = ShapeType.CIRCLE
            geo.pixel_size = radius * 2
            geo.rotation_angle = 0
            geo.valid = True

        return geo

    def filter_by_size(self, geometries: List[GeometryInfo],
                       size_range: Tuple[float, float]) -> List[GeometryInfo]:
        """尺寸过滤"""
        """
        geometries: 存放GeometryInfo类对象的列表
        size_range: 尺寸范围
        List[GeometryInfo]: 返回存放GeometryInfo类对象的列表
        """
        return [g for g in geometries if size_range[0] <= g.size <= size_range[1]]


class DistanceCalculator:
    """距离计算器"""

    A4_WIDTH = 210.0
    MIN_DISTANCE, MAX_DISTANCE = 800, 1500

    def __init__(self):
        # 后面会通过标定自动算出正确值，这里只是默认值
        self.focal_length = 800.0        # 焦距
        self.mm_per_pixel_at_1m = 0.5    # 1米远处，1像素=多少毫米

    def calibrate(self, known_distance_mm: float, pixel_width: float):
        """标定"""
        """
        known_distance_mm: 已知的距离
        pixel_width: 像素宽度
        """
        # 焦距：镜头中心到镜面的距离
        self.focal_length = (pixel_width * known_distance_mm) / self.A4_WIDTH
        self.mm_per_pixel_at_1m = self.A4_WIDTH / pixel_width
        print(f"标定: 焦距={self.focal_length:.2f}px, 像素当量={self.mm_per_pixel_at_1m:.4f}mm/px")

    def calculate_distance(self, pixel_width: float) -> float:
        """计算距离"""
        """
        pixel_width: 像素宽度
        返回值: 距离(mm)
        """
        if pixel_width <= 0:
            return 0.0
        # 距离 = (真实宽度 × 焦距) ÷ 图像像素宽度
        distance = (self.A4_WIDTH * self.focal_length) / pixel_width
        return float(max(self.MIN_DISTANCE, min(int(distance), self.MAX_DISTANCE)))

    def pixel_to_mm(self, pixel_size: float, distance_mm: float) -> float:
        """像素转毫米"""
        """
        pixel_size: 边长宽度（像素）
        distance_mm: 实际距离（mm）
        返回值: 实际距离（mm）
        """
        # 自动根据 “实际距离” 修正的公式
        return pixel_size * self.mm_per_pixel_at_1m * (distance_mm / 1000.0)


class MeasurementSystem:
    """测量系统主类"""

    WARPED_SIZE = (400, 565)    # 这个是A4纸的比例

    def __init__(self):
        # 1. 初始化所有工具，创建类对象
        self.a4_detector = A4BorderDetector()    # A4检测器
        self.geo_detector = GeometryDetector()   # 图形检测器
        self.calculator = DistanceCalculator()   # 距离计算器

        # 2. 状态标志
        self.is_calibrated = False               # 是否完成标定（一开始没标定）
        self.mode = "basic"                      # 默认模式：基本模式

        # 3. 初始化屏幕
        self.lcd = SPILCD()

        # 4. 打开摄像头，设置分辨率640x480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 5. 初始化按键，绑定回调函数
        # 将函数作为参数传入
        self.buttons = GPIOButtons(
            measure_callback=self._on_measure,  # 按测量键 → 执行 _on_measure
            mode_callback=self._on_mode         # 按模式键 → 执行 _on_mode
        )

        print("系统初始化完成")
        print("按GPIO17启动测量，按GPIO27切换模式")

    def _on_measure(self):
        """测量按键回调"""
        if not self.is_calibrated:
            print("请先标定")
            return

        # 拍一张照片
        ret, frame = self.cap.read()
        if ret:
            # 执行核心测量逻辑
            result = self._measure(frame, self.mode == "advanced")
            # 把结果显示到屏幕
            self.lcd.display_result(result, self.mode)
            if result.valid:
                print(f"D={result.distance:.1f}mm, x={result.geometry.size:.1f}mm")

    def _on_mode(self):
        """模式按键回调"""
        self.mode = "advanced" if self.mode == "basic" else "basic"
        print(f"切换到{'发挥' if self.mode == 'advanced' else '基本'}模式")

    def calibrate(self) -> bool:
        """标定"""
        print("请在100cm距离放置A4纸，按测量键开始标定")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 检测A4
            border_info = self.a4_detector.detect_a4_border(frame)
            if border_info:
                self.calculator.calibrate(1000.0, border_info['pixel_width'])
                self.is_calibrated = True
                print("标定成功!")

                result = MeasurementResult(valid=True, distance=1000.0)
                self.lcd.display_result(result)
                return True

            time.sleep(0.1)

    def _get_warped_image(self, image: np.ndarray, border_info: Dict) -> np.ndarray:
        """透视变换获取校正后的图像"""
        """
        image:原始摄像头图像
        border_info:角点信息
        np.ndarray:一张经透视变换矫正后的图（比例是A4纸的比列）
        """
        # 目标四个点（矩形）
        dst_points = np.array([
            [0, 0],                                         # 左上角
            [self.WARPED_SIZE[0], 0],                       # 右上角
            [self.WARPED_SIZE[0], self.WARPED_SIZE[1]],     # 右下角
            [0, self.WARPED_SIZE[1]]                        # 左下角
        ], dtype=np.float32)

        # 计算 “变形矩阵”
        transform_matrix = cv2.getPerspectiveTransform(border_info['corners'],  # 原来的4个角点（歪的A4）
                                                       dst_points)              # 目标4个角点（正的矩形）
        # 执行变换，把A4拍平
        return cv2.warpPerspective(image,               # 原始图片
                                   transform_matrix,    # 变换矩阵（怎么变）
                                   self.WARPED_SIZE     # 输出图片大小
                                   )

    def _measure(self, image: np.ndarray, is_advanced: bool) -> MeasurementResult:
        """统一测量方法"""
        """
        image: 原始图像
        is_advanced: 是否是发挥模式
        返回值: 一个测量结果类
        """
        # 新建结果对象
        result = MeasurementResult()
        # 1. 找A4
        border_info = self.a4_detector.detect_a4_border(image)
        if border_info is None:
            return result
        # 2. 计算距离
        result.distance = self.calculator.calculate_distance(border_info['pixel_width'])
        # 3. 把A4拍平
        warped = self._get_warped_image(image, border_info)
        # 4. 识别所有图形
        geometries = self.geo_detector.detect_all_geometries(warped, is_advanced)

        # 转换像素尺寸为毫米
        for geo in geometries:
            geo.size = self.calculator.pixel_to_mm(geo.pixel_size, result.distance)
        # ====================================== 模式选择 ======================================
        if is_advanced:
            # 发挥模式：找最小正方形， 最小旋转角度
            geometries = self.geo_detector.filter_by_size(geometries,
                                                          self.geo_detector.ADVANCED_SIZE_RANGE)
            # 从所有图形里，只挑出正方形，放入列表中
            squares = [g for g in geometries if g.shape_type == ShapeType.SQUARE]
            if squares:
                min_square = min(squares, key=lambda g: g.size)
                result.geometry = min_square
                result.min_size = min_square.size
                result.min_angle = min_square.rotation_angle
            elif geometries:
                result.geometry = geometries[0]
        else:
            # 基本模式：找最大图形
            geometries = self.geo_detector.filter_by_size(geometries,
                                                          self.geo_detector.BASIC_SIZE_RANGE)
            if geometries:
                result.geometry = max(geometries, key=lambda g: g.pixel_size)
                if result.geometry.shape_type == ShapeType.CIRCLE:
                    result.geometry.rotation_angle = 0

        result.valid = bool(geometries)  # 有图形就是有效
        return result

    def run(self):   # 主运行函数
        """主循环"""
        print("=" * 60)
        print("单目视觉目标物测量系统")
        print("=" * 60)

        self.calibrate()    # 开机先标定！

        print("\n系统运行中...")
        print("GPIO17: 测量")
        print("GPIO27: 切换模式")

        # 保持程序运行
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n系统停止")
        finally:
            self.buttons.cleanup()
            self.cap.release()


def main():
    system = MeasurementSystem()    # ① 创建系统
    system.run()                    # ② 运行系统


if __name__ == "__main__":
    main()
