class BBoxGrouping:
    def __init__(self, bboxes, iou_threshold):
        """
        Initializes the BBoxGrouping object.

        :param bboxes: A list of BBox objects.
        :param iou_threshold: A threshold for IoU above which two bounding boxes are considered to mark the same object.
        """
        self.bboxes = bboxes
        self.iou_threshold = iou_threshold
        self.similarity_matrix = [[0.0 for _ in range(len(bboxes))] for _ in range(len(bboxes))]
        self.groups = []

    def calculate_similarity_matrix(self):
        """
        Calculates and populates the similarity matrix with IoU values between each pair of bounding boxes.
        """
        for i in range(len(self.bboxes)):
            for j in range(len(self.bboxes)):
                if i != j:
                    # Calculate IoU between bbox[i] and bbox[j]
                    iou = self.bboxes[i].calculate_iou(self.bboxes[j])
                    self.similarity_matrix[i][j] = iou

    def group_bboxes(self):
        """
        Groups bounding boxes based on the similarity matrix and IoU threshold,
        preferring to group with the bbox that has the most overlap.
        """
        # Initialize a list to track which bounding boxes have been grouped
        grouped = [False] * len(self.bboxes)

        # First, assign standalone bounding boxes to their own groups
        for i in range(len(self.bboxes)):
            if not any(self.similarity_matrix[i][j] > 0 for j in range(len(self.bboxes)) if i != j):
                self.groups.append([i])
                grouped[i] = True

        # Now, group the remaining bounding boxes
        for i in range(len(self.bboxes)):
            if grouped[i]:
                continue

            current_group = [i]
            grouped[i] = True

            for j in range(len(self.bboxes)):
                if not grouped[j] and any(self.similarity_matrix[k][j] >= self.iou_threshold for k in current_group):
                    current_group.append(j)
                    grouped[j] = True

            self.groups.append(current_group)
