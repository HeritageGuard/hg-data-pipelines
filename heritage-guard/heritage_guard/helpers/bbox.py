class BBox:
    def __init__(self, bbox, score, object_class):
        """
    Initializes a BBox object and converts the Cartesian coordinates to spherical coordinates.

    :param bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box in Cartesian coordinates.
    """
        self.image_width = 8000
        self.image_height = 4000
        self.score = score
        self.object_class = object_class
        # Check if bounding box crosses the seam
        bbox_span = bbox[2] - bbox[0]
        if bbox_span > self.image_width * 0.95:  # 95% threshold can be adjusted
            self.x_max, self.y_min, self.x_min, self.y_max = bbox
        else:
            self.x_min, self.y_min, self.x_max, self.y_max = bbox

        self.theta_min, self.theta_max = self.convert_bbox_to_cylindrical(self.x_min, self.x_max)

    def convert_bbox_to_cylindrical(self, x_min, x_max):
        theta_min = (x_min / self.image_width) * 360
        theta_max = (x_max / self.image_width) * 360

        return theta_min, theta_max

    @property
    def width(self) -> int:
        return (self.x_max - self.x_min) % self.image_width

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @staticmethod
    def overlap(min1, max1, min2, max2):
        return max(0, min(max1, max2) - max(min1, min2))

    @staticmethod
    def calculate_theta_overlap(theta_min1, theta_max1, theta_min2, theta_max2):
        def normalize_theta(theta):
            return theta % 360

        theta_min1, theta_max1 = normalize_theta(theta_min1), normalize_theta(theta_max1)
        theta_min2, theta_max2 = normalize_theta(theta_min2), normalize_theta(theta_max2)

        crosses_seam1 = theta_max1 < theta_min1
        crosses_seam2 = theta_max2 < theta_min2

        if not crosses_seam1 and not crosses_seam2:
            # Standard overlap calculation when neither bbox crosses the seam
            return BBox.overlap(theta_min1, theta_max1, theta_min2, theta_max2)
        else:
            # Adjust theta values for seam-crossing bboxes
            if crosses_seam1:
                theta_max1 += 360
            if crosses_seam2:
                theta_max2 += 360

            # Calculate overlap considering adjusted theta values
            overlap = BBox.overlap(theta_min1, theta_max1, theta_min2, theta_max2)

            # If overlap is greater than 360, it means we're counting the non-overlapping part
            return min(overlap, 360)

    def calculate_iou(self, other):
        theta_overlap = BBox.calculate_theta_overlap(self.theta_min, self.theta_max, other.theta_min, other.theta_max)
        y_overlap = BBox.overlap(self.y_min, self.y_max, other.y_min, other.y_max)

        intersection_area = theta_overlap * y_overlap
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0  # Avoid division by zero

        iou = intersection_area / union_area
        return iou
