import  cv2
import numpy as np
import math
import timeit

class Area(object):
  pixel: [int, int, int]
  points: [(int, int)]

  def __init__(self, x, y, pixel = [0, 0, 0]):
    self.x = x
    self.y = y
    self.pixel = pixel
    self.points = []
  
  def update(self, x, y, pixel):
    self.x = x
    self.y = y
    self.pixel = pixel

class Image(object):
  @staticmethod
  def open(fileName):
    image = cv2.imread(fileName)
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  
  @staticmethod
  def save(fileName, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(fileName, rgb_image)
  
class SLIC(object):
  def __init__(self, fileName, num):
    self.image = Image.open(fileName)
    print(self.image.shape)
    self.height = self.image.shape[0]
    self.width = self.image.shape[1]
    self.N = self.height * self.width
    self.num = num #超像素个数
    self.dis = int(math.sqrt(self.N / self.num))
    self.areas = []
    self.label = {}
    self.distances = np.full((self.height, self.width), np.inf)
    self.gradients = np.zeros((self.height, self.width))
  
  def init_areas(self):
    i = self.dis // 2
    j = self.dis // 2
    while i < self.height:
      while j < self.width:
        self.areas.append(Area(i, j, self.image[i, j]))
        j += self.dis
      j = self.dis // 2
      i += self.dis

  def get_gradients(self):
    for j in range(0, self.width - 1):
      self.gradients[:, j] = np.sum(self.image[:, j, :] - self.image[:, j + 1, :], axis=1)
    self.gradients[:, self.width - 1] = self.gradients[:, self.width - 2]

  def adjust(self):
    for area in self.areas:
      curren_gra = self.gradients[area.x, area.y]
      for i in range(-1, 2):
        for j in range(-1, 2):
          x = area.x + i
          y = area.y + j
          if x < 0 or x >= self.height or y < 0 or y >= self.width or (i == 0 and j == 0):
            continue
          if self.gradients[x, y] < curren_gra:
            area.update(x, y, self.image[x, y])
            curren_gra = self.gradients[x, y]
  
  def search(self):
    start = timeit.default_timer()
    for area in self.areas:
      for i in range(area.x - 2 * self.dis, area.x + 2 * self.dis):
        if i < 0 or i >= self.height:
          continue
        for j in range(area.y - 2 * self.dis, area.y + 2 * self.dis):
          if j < 0 or j >= self.width:
            continue
          L, A, B = self.image[i][j]
          Dc = math.sqrt(
            math.pow(int(L) - area.pixel[0], 2) +
            math.pow(int(A) - area.pixel[1], 2) +
            math.pow(int(B) - area.pixel[2], 2))
          Ds = math.sqrt(
            math.pow(i - area.x, 2) +
            math.pow(j - area.y, 2))
          D = math.sqrt(math.pow(Dc / 32, 2) + math.pow(Ds / self.dis, 2))
          if D < self.distances[i][j]:
            if (i, j) not in self.label:
              self.label[(i, j)] = area
            else:
              self.label[(i, j)].points.remove((i, j))
              self.label[(i, j)] = area
            area.points.append((i, j))
            self.distances[i, j] = D
    print('done ' + str(timeit.default_timer() - start))

  def update(self):
    for area in self.areas:
      sum_i = sum_j = 0
      for p in area.points:
        sum_i += p[0]
        sum_j += p[1]
      i = sum_i // len(area.points)
      j =  sum_j // len(area.points)
      area.update(i, j, self.image[i, j])
  
  def train(self):
    self.init_areas()
    self.get_gradients()
    self.adjust()
    for i in range(10):
      self.search()
      self.update()
      name = 'lenna_loop{loop}.png'.format(loop=i)
      self.save(name)
    
  def save(self, fileName):
    image = np.copy(self.image)
    for area in self.areas:
      for p in area.points:
        image[p[0], p[1]] = area.pixel
    Image.save(fileName, image)

slic = SLIC('timg.jpg', 1024)
slic.train()
    
