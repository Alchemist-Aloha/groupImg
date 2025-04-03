import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore")


class K_means:
    def __init__(self, k=3, size=False, resample=32, color_weight=1.0, size_weight=0.5):
        self.k = k
        self.cluster = []
        self.data = []
        self.end = []
        self.i = 0
        self.size = size
        self.resample = resample
        self.color_weight = color_weight
        self.size_weight = size_weight

    def manhattan_distance(self, x1, x2):
        s = 0.0
        # Apply weights to different feature types
        feature_count = len(x1)
        size_features = 3 if self.size else 0
        color_features = feature_count - size_features

        # Weight color histogram features
        for i in range(color_features):
            s += abs(float(x1[i]) - float(x2[i])) * self.color_weight

        # Weight size features
        if self.size:
            for i in range(color_features, feature_count):
                s += abs(float(x1[i]) - float(x2[i])) * self.size_weight

        return s

    def euclidian_distance(self, x1, x2):
        s = 0.0
        for i in range(len(x1)):
            s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
        return s

    def read_image(self, im):
        if self.i >= self.k:
            self.i = 0
        try:
            img = Image.open(im)
            osize = img.size
            img.thumbnail((self.resample, self.resample))

            # Extract more features - RGB channels separately
            img_array = np.asarray(img)
            features = []

            # Add histogram features for each channel if it's a color image
            if len(img_array.shape) > 2 and img_array.shape[2] >= 3:
                for channel in range(3):  # RGB
                    channel_hist = np.histogram(
                        img_array[:, :, channel], bins=8, range=(0, 256)
                    )[0]
                    # Normalize
                    channel_hist = channel_hist / (img.size[0] * img.size[1]) * 100
                    features.extend(channel_hist)
            else:
                # Grayscale image
                hist = np.histogram(img_array, bins=16, range=(0, 256))[0]
                hist = hist / (img.size[0] * img.size[1]) * 100
                features.extend(hist)

            # Add image size features if requested
            if self.size:
                # Add size ratio as a feature
                aspect_ratio = osize[0] / osize[1] if osize[1] > 0 else 0
                features.extend([osize[0], osize[1], aspect_ratio])

            pbar.update(1)
            i = self.i
            self.i += 1
            return [i, features, im]
        except Exception as e:
            print("Error reading ", im, e)
            return [None, None, None]

    def generate_k_means(self):
        final_mean = []
        for c in range(self.k):
            partial_mean = []
            for i in range(len(self.data[0])):
                s = 0.0
                t = 0
                for j in range(len(self.data)):
                    if self.cluster[j] == c:
                        s += self.data[j][i]
                        t += 1
                if t != 0:
                    partial_mean.append(float(s) / float(t))
                else:
                    partial_mean.append(float("inf"))
            final_mean.append(partial_mean)
        return final_mean

    def generate_k_clusters(self, folder):
        pool = ThreadPool(cpu_count())
        result = pool.map(self.read_image, folder)
        pool.close()
        pool.join()
        self.cluster = [r[0] for r in result if r[0] != None]
        self.data = [r[1] for r in result if r[1] != None]
        self.end = [r[2] for r in result if r[2] != None]

    def initialize_clusters(self):
        """K-means++ initialization for better starting clusters"""
        # Choose first centroid randomly
        first_idx = np.random.randint(0, len(self.data))
        centroids_idx = [first_idx]
        centroids = [self.data[first_idx]]

        # Choose remaining centroids
        for _ in range(1, self.k):
            # Calculate distances to nearest centroid for each point
            distances = []
            for i in range(len(self.data)):
                if i in centroids_idx:
                    distances.append(0)
                    continue

                # Find minimum distance to any existing centroid
                min_dist = float("inf")
                for cent in centroids:
                    dist = self.euclidian_distance(self.data[i], cent)
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)

            # Choose next centroid with probability proportional to distance squared
            distances = np.array(distances)
            probabilities = distances**2 / np.sum(distances**2)
            next_idx = np.random.choice(len(self.data), p=probabilities)
            centroids_idx.append(next_idx)
            centroids.append(self.data[next_idx])

        # Set initial clusters based on the selected centroids
        for i in range(len(self.data)):
            min_dist = float("inf")
            best_cluster = 0
            for c, centroid in enumerate(centroids):
                dist = self.euclidian_distance(self.data[i], centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = c
            self.cluster[i] = best_cluster

    def rearrange_clusters(self):
        isover = False
        while not isover:
            isover = True
            m = self.generate_k_means()
            for x in range(len(self.cluster)):
                dist = []
                for a in range(self.k):
                    # Use Euclidean distance instead of Manhattan
                    dist.append(self.euclidian_distance(self.data[x], m[a]))
                _mindist = dist.index(min(dist))
                if self.cluster[x] != _mindist:
                    self.cluster[x] = _mindist
                    isover = False


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to image folder")
ap.add_argument("-k", "--kmeans", type=int, default=5, help="how many groups")
ap.add_argument(
    "-r", "--resample", type=int, default=128, help="size to resample the image by"
)
ap.add_argument(
    "-s",
    "--size",
    default=False,
    action="store_true",
    help="use size to compare images",
)
ap.add_argument(
    "-m", "--move", default=False, action="store_true", help="move instead of copy"
)
ap.add_argument(
    "--color-weight", type=float, default=1.0, help="weight for color features"
)
ap.add_argument(
    "--size-weight", type=float, default=0.5, help="weight for size features"
)
ap.add_argument(
    "--iterations", type=int, default=10, help="maximum iterations for k-means"
)
ap.add_argument(
    "--channels", type=int, default=3, help="number of color channels to use (1 or 3)"
)
args = vars(ap.parse_args())
types = (
    "*.jpg",
    "*.JPG",
    "*.png",
    "*.jpeg",
    "*.PNG",
    "*.JPEG",
    "*.tiff",
    "*.TIFF",
    "*.webp",
    "*.WEBP",
)
imagePaths = []
folder = args["folder"]
if not folder.endswith("/"):
    folder += "/"
for files in types:
    imagePaths.extend(sorted(glob.glob(folder + files)))
nimages = len(imagePaths)
nfolders = int(math.log(args["kmeans"], 10)) + 1
if nimages <= 0:
    print("No images found!")
    exit()
if args["resample"] < 16 or args["resample"] > 256:
    print("-r should be a value between 16 and 256")
    exit()
pbar = tqdm(total=nimages)
k = K_means(args["kmeans"], args["size"], args["resample"], args["color_weight"], args["size_weight"])
k.generate_k_clusters(imagePaths)
k.initialize_clusters()
k.rearrange_clusters()
for i in range(k.k):
    try:
        os.makedirs(folder + str(i + 1).zfill(nfolders))
    except FileExistsError:
        print("Folder already exists")
action = shutil.copy
if args["move"]:
    action = shutil.move
for i in range(len(k.cluster)):
    action(k.end[i], folder + "/" + str(k.cluster[i] + 1).zfill(nfolders) + "/")
