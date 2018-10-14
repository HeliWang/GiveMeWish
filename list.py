from collections import Counter

"""
1. Given an array and a target, return the number of pairs whose sum is the target.
10/12/2018 21:00

Idea: For special cases, list example and calculate the correct answer by math.

https://www.geeksforgeeks.org/count-pairs-with-given-sum/
"""

# [1, 2, 4, 5, 3, 1, 2, 0] t = 4 => (4, 0), (3, 1), (3, 1), (2, 2)

print("1. count_pair (dict)")

def count_pair(nums, t):
    count_dict = Counter(nums)

    # say t = 4, there are 5 times of 2
    # first 2, pair with other 4 times of 2
    # second 2, pair with other 3 times of 2
    # ....
    # (5 - 1) + .... + 0
    # (5 - 1) * 5 // 2
    res = 0
    for n in nums:
        res += count_dict[t - n]
        if 2 * n == t:
            res -= 1
    return res // 2

print(count_pair([1, 2, 4, 5, 3, 1, 2, 0], 4))

print()

"""
2. Given nums and v, return index i where i is the largest index for nums[i] <= v
"""

print("2. get_lower_bound")

def get_lower_bound(nums, v):
    # return index i where i is the smallest index for nums[i] >= v
    # just use l

    # nums is sorted
    # [1, 2, 3, 4, 5]
    # v = 3 -->
    #  l = 2 (pointing 3 which is just >= v),
    #  r = 1 (pointing 2, which is pointing the last element < v)

    # [1, 2, 3, 3, 5]
    # v = 3 -->
    #  l = 2 (pointing 3 which is just >= v),
    #  r = 1 (pointing 2, which is pointing the last element < v)

    # [1, 2, 3, 3, 5]
    # v = 4
    #  l = 4 (pointing 5 which is just > v),
    #  r = 3 (pointing 3, which is pointing the last element < v)

    # [1, 2, 3, 3, 5]
    # v = 6
    #  l = 5 (OUT OF BOUND),
    #  r = 4 (pointing 3, which is pointing the last element < v)

    # [1, 2, 3, 3, 5]
    # v = 0
    #  l = 0 (pointing 5 which is just >= v),
    #  r = -1 (OUT OF BOUND)

    # [1, 2, 3, 3, 5]
    # v = 1
    #  l = 0 (pointing 2 which is just >= v),
    #  r = -1 (OUTOFBOUND, which is pointing the last element < v)

    # [1, 2, 3, 3, 5]
    # v = 5
    #  l = 4
    #  r = 3 (which is pointing the last element < v)

    n = len(nums)
    l = 0
    r = n - 1
    while l <= r:
        mid = (l + r) // 2
        # mid keeps increasing, 0 <= mid <= n - 1
        if nums[mid] < v: # !!! <
            l = mid + 1
        else:
            r = mid - 1
    return l, r

print(get_lower_bound([1, 2, 3, 4, 5], 3))
print(get_lower_bound([1, 2, 3, 3, 5], 3))
print(get_lower_bound([1, 2, 3, 3, 5], 4))
print(get_lower_bound([1, 2, 3, 3, 5], 6))
print(get_lower_bound([1, 2, 3, 3, 5], 0))
print(get_lower_bound([1, 2, 3, 3, 5], 1))
print(get_lower_bound([1, 2, 3, 3, 5], 5))

print()
"""
2. Given nums and v, return index i where i is the largest index for nums[i] <= v
"""

print("3. get_upper_bound")
def get_upper_bound(nums, v):
    # return index i where i is the largest index for nums[i] <= v
    # just use r

    # nums is sorted
    # [1, 2, 3, 4, 5]
    # v = 3 -->
    #  l = 3 (pointing 4 which is just > v),
    #  r = 2 (pointing 3, which is pointing the last element <= v)

    # [1, 2, 3, 3, 5]
    # v = 3 -->
    #  l = 4 (pointing 5 which is just > v),
    #  r = 3 (pointing 3, which is pointing the last element <= v)

    # [1, 2, 3, 3, 5]
    # v = 4
    #  l = 4 (pointing 5 which is just > v),
    #  r = 3 (pointing 3, which is pointing the last element <= v)

    # [1, 2, 3, 3, 5]
    # v = 6
    #  l = 5 (OUT OF BOUND),
    #  r = 4 (pointing 3, which is pointing the last element <= v)

    # [1, 2, 3, 3, 5]
    # v = 0
    #  l = 0 (pointing 5 which is just > v),
    #  r = -1 (OUT OF BOUND)

    # [1, 2, 3, 3, 5]
    # v = 1
    #  l = 1 (pointing 2 which is just > v),
    #  r = 0

    # [1, 2, 3, 3, 5]
    # v = 5
    #  l = 5
    #  r = 4

    n = len(nums)
    l = 0
    r = n - 1
    while l <= r:
        mid = (l + r) // 2
        # mid keeps increasing, 0 <= mid <= n - 1
        if nums[mid] <= v: #!!! <=
            l = mid + 1
        else:
            r = mid - 1
    return l, r

print(get_upper_bound([1, 2, 3, 4, 5], 3))
print(get_upper_bound([1, 2, 3, 3, 5], 3))
print(get_upper_bound([1, 2, 3, 3, 5], 4))
print(get_upper_bound([1, 2, 3, 3, 5], 6))
print(get_upper_bound([1, 2, 3, 3, 5], 0))
print(get_upper_bound([1, 2, 3, 3, 5], 1))
print(get_upper_bound([1, 2, 3, 3, 5], 5))

print()

"""
4. Binary Search binary_search (nums, target)
"""

def binary_search(nums, target):
    n = len(nums)
    l = 0
    r = n - 1 # use the get_upper_bound template
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid - 1
    if r < 0 or nums[r] != target: return -1 # !!!!!!
    return r

print("4. binary_search")
print()

"""
5. return [start, end] such that if the range is sorted, then the whole array is sorted. 
[1, 2, 3, 5, 4, 6, 7] => [3, 4] 
[1, 2, 3, 6, 1, 2, 5] => [1, 6]
[1, 2, 3, 6, 3, 2, 5, 7, 6] => [2, 8]
https://www.geeksforgeeks.org/minimum-length-unsorted-subarray-sorting-which-makes-the-complete-array-sorted/

Idea: Base Case -> Improve Time Complexity

What if not exists -- -1

Base Solution: O(n*logn) by sorting
"""
# get_range([1, 2, 3, 5, 4, 6, 7, 8, 9, 7, 10])  # [2, 4]

print("5. get_range")

def get_range(nums):
    n = len(nums)
    range_l = -1
    range_r = -1
    max_sofar = nums[0]

    for i, num in enumerate(nums):
        if num < max_sofar:
            # unsorted starting at i
            if range_l == -1:
                range_l = get_upper_bound(nums, min(nums[i:]))[0]
            range_r = i
        else:
            max_sofar = num
    return range_l, range_r

print(get_range([1, 2, 3, 5, 4, 6, 7]))
print(get_range([1, 2, 3, 6, 1, 2, 5]))
print(get_range([1, 2, 3, 6, 3, 2, 5, 7, 6]))
print(get_range([10, 12, 20, 30, 25, 40, 32, 31, 35, 50, 60])) # (3. 8)

print()


"""
6. Find the integer x appears more than n/k times in a sorted array of n integers. 
Return None if not exists

Test Cases Build:
[1, 2, 3, 4]
[1, 2, 3, 4, 5, 6, 6]
[1, 2, 2, 4, 5, 6, 6]
[1, 2, 2, 4, 6, 6, 7]

Found out there might be multiple solutions -> return a list of solutions
1. 对于满足条件的数，一定只会出现在0, 1 * length / k, 2 * length / k, 3 * length / k, ... , min( k * length / k, length - 1), 注意最后的边界要取一个较小值。
2. 对于条件2中的数字，利用二分法找到第一个出现的位置和最后一个出现的位置，只要次数 >= length / k，就加入到结果中。

s = n // 4

Utilize the properity "sorted" to help solving the problem.
--> What difference will make if the array is sorted.

"""

print("6. find_popular (binary search)")


def find_first(items, i, j, n):
    # find the first idx of occurance
    first = -1
    while i <= j:
        mid = (i + j) >> 1
        if items[mid] < n:
            i = mid + 1
        elif items[mid] == n:
            first = mid
            j = mid - 1
        else:
            j = mid - 1
    return first


def find_last(items, i, j, n):
    # find the last idx of occurance
    last = -1
    while i <= j:
        mid = (i + j) >> 1
        if items[mid] < n:
            i = mid + 1
        elif items[mid] == n:
            last = mid
            i = mid + 1
        else:
            j = mid - 1
    return last

# First Implementation: find at most 2*k points
# 2*k*log(n)

def find_popular(nums, k):
    n = len(nums)
    res_set = set()
    freq = n // k
    bounds = [i * freq for i in range(2 * k) if i * freq <= n - 1]
    # add right-most boundary
    # bounds.append(n - 1)
    candidates = [nums[i] for i in bounds]
    for i in range(1, len(candidates) - 1):
        first = find_first(nums, bounds[i - 1], bounds[i], candidates[i])
        last = find_last(nums, bounds[i], bounds[i + 1], candidates[i])
        # because boundaries are included
        # candidate must be found
        if last - first + 1 > freq:
            # we use set because candidates may be the same
            res_set.add(candidates[i])
    return res_set

print(find_popular([1, 2, 2, 4, 6, 6, 7], 4)) # {2, 6}
print(find_popular([1, 2, 2, 4, 6, 7, 7], 4)) # {2, 7}


# Second Implementation: find at most 2*k points
# 2*k*(2*k)
# For each point, look around the k before and k after points.

print()

"""
Merge sort without recursion
"""

print("7. Iterative Merge Sort")

print()


"""
Merge sort without recursion
"""

print("8. Rotate Matrix (matrix)")
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        for row in range(len(matrix) / 2):
            for column in range((len(matrix) + 1)/ 2):
                a = matrix[row][column]
                b = matrix[column][len(matrix) - 1 - row]
                c = matrix[len(matrix) - 1 - row][len(matrix) - 1 - column]
                d = matrix[len(matrix) - 1 - column][row]
                matrix[row][column] = d
                matrix[column][len(matrix) - 1 - row] = a
                matrix[len(matrix) - 1 - row][len(matrix) - 1 - column] = b
                matrix[len(matrix) - 1 - column][row] = c

print()

print("9. K Inverse Pairs Array (DP)")

def kInversePairs(n, k):
    """
    n = 1
    [1]

    n = 2
    [1, 2] [2, 1]
        k = 1 [2, 1]

    n = 3
    [1, 2, 3] [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]
        k = 1  [1,3,2]   and [2,1,3]
        k = 1  [3, 1, 2] and [2,3,1]


    n = 4
        k = 3
    [3, 2, 1. 4], [3, 1, 4, 2], [2, 3, 4, 1],  [1, 4, 3, 2]. [2, 4, 1, 3], [4, 1, 2, 3],
    [4, 1, 3, 2], [4, 3, 1, 2] (5 pairs), [4, 3, 2, 1] (6 pairs)
    kInversePairs(self, n, k) =
        kInversePairs(n, k - 1) + kInversePairs(n - 1, k)

    P(n,r) = n! / (n−r)!
    C(n,r) = n! / ((n−r)! * r!)

    """
    states = [1] + [0] * k # for n = 0, [0, k] pairs
    for i in range(1, n + 1):
        next_states = [1] + [0] * k
        for j in range(1, k + 1):
            next_states[j] = next_states[j - 1] + states[j]
            if j - 1 - (i - 1) >= 0:
                next_states[j] -= states[j - 1 - (i - 1)]
        states = next_states
    return states[k] % (10 ** 9 + 7)

def kInversePairsComplex(n, k):
    """
    :type n: int
    :type k: int
    :rtype: int
    """
    max_k = n * (n - 1) // 2
    if k > max_k: return 0
    # states = [[0 for _ in range(max_k + 1)] for _ in range(n + 1)]
    states = [[0 for _ in range(max_k + 1)] for _ in range(2)]

    cur_row = 0
    states[0][0] = 1

    for i in range(1, n + 1):
        new_row = (cur_row + 1) % 2
        states[new_row][0] = 1
        for j in range(1, min(i * (i - 1) // 2 + 1, k + 1)):
            # states[new_row][j] = sum([states[cur_row][k] for k in range((j - i + 1), j + 1)])
            states[new_row][j] = states[new_row][j - 1] - states[cur_row][j - 1 - (i - 1)] + states[cur_row][j]
        cur_row = new_row
    return states[cur_row][k] % (10 ** 9 + 7)

print()

print("10. Longest Consecutive Sequence (Set / Union Find)")

from collections import Counter

class Solution:
    def longestConsecutive(self, nums):
        nums_set = set(nums)
        res = 0
        for num in nums:
            cluster_size = 0
            n = num
            while True:
                if n in nums_set:
                    nums_set.remove(n)
                else:
                    break
                cluster_size += 1
                n -= 1
            n = num + 1
            while True:
                if n in nums_set:
                    nums_set.remove(n)
                else:
                    break
                cluster_size += 1
                n += 1
            res = max(res, cluster_size)
        return res

    """
    Complexity of union-find with path-compression, without rank. 
    Union by rank without path compression
         gives an amortized time complexity of O (log n)
    Union by rank with path compression
         gives an amortized time complexity of O(1) < < O(logn)
    """

    def find(self, x):
        y = x
        while self.parents[y] != y:
            y = self.parents[y]
        self.parents[x] = y
        return y

    def union(self, x, y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        if parent_x != parent_y:
            self.parents[parent_x] = parent_y

    def longestConsecutiveUnionFind(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = list(set(nums))
        self.parents = dict(zip(nums, nums))
        for num in nums:
            if num - 1 in self.parents:
                self.union(num - 1, num)
            if num + 1 in self.parents:
                self.union(num + 1, num)
        count = Counter([self.find(n) for n in nums])
        return max(count.values() or [0])

print()

"""
给出一个10K的文档，里面很多文章。现在给出一个String query, 找到最有关系的10个文章。
有很多文章，实现一个很简单的搜索引擎，每篇文章对应一个word count，然后根据word去找
Step1: 第一步是在索引中找到包含这三个词的网页
Step2: 概括地讲，如果一个查询包含关键词 w1,w2,...,wN, 计算上述文档中，查询词的TF-IDF之和

给一个时间，比如9:30，返回时针和分针的夹角
https://blog.csdn.net/prstaxy/article/details/22210829

3Sum / Subarray Sum Equals K

题目是给一些关系，比如 A 和 B的关系， B和 C的关系... 然后给一个 start 和target， print 出所有可以从start 出发，
在target截止，并且带上 relationship. 比如 A brother B, B mother C,  B mother A, B friend D.  给 start = A, 
target = C 的话， 要print 出 A brother B, B mother C。

设计一个21点的赌博游戏。 游戏里有1个庄家，k个玩家，玩法是 每个玩家刚开始有2张牌，面朝上，庄家2张牌，一张面朝上。
 从第一个玩家开始发牌，可以选择hit/pass. 如果hit的话就加分，但如果超过21点就输（可以选择很多次，如果玩家和庄家都没有超过21点， 
 比谁的分大。 最后print 出每个玩家赢还是庄家赢。主要考点是OOD，具体怎么实现不是特别care, 比如洗牌什么的只问了用什么data structure
  好之类的。

给出2个数组，比如A =[1,6,9], B = [1,1,1]. 每个数字相加，但是保证每个slot 的值小于10， 如果大于10 就要分开。
比如刚才2个A, B 的结果应该输出 [2,7,1,0];

加油站问题，给出Gas, Cost 2组数组。输出所有可以开头并且走完一圈的index.

design 一个train。train在循环的铁路上面跑。问题分三部分
（1）开始要求两个function advance和getcur。 advance一次前进一站，getcur 返回当前车站
（2）follow up 加一个request function， advance只能跑到距离最近的被request的车站
（3）follow up pick up乘客，给上车和下车。

旋转一个链表，就是把前面几个append到最后去，先写2pass然后1pass怎么办，用两个指针就行了

Find the longest path in a matrix with given constraints
https://www.geeksforgeeks.org/find-the-longest-path-in-a-matrix-with-given-constraints/

DRAW A DIAMOND WITH ASTERISKS USING RECURSION
http://code.activestate.com/recipes/578959-draw-a-diamond-with-asterisks-using-recursion/
要求最好是递归,但是迭代也可以

Maximum Subarray 红绿灯，每次可以toggle连续一个区间的灯（红变绿，绿变红），问得到最多绿灯，怎么办，其实就是lc53，1和-1的数组
交通控制。给出一个Char Array, 里面只有 R, G. 现在要选择一个范围，
使得 R 变成G，G变成R，并使得G的个数－ R的个数最大。输出这个范围并且输出最大结果

timestamp + uid 找bot
限制某个user对api的访问次数，如果多余一个阈值，就认为是一个bot
A bot is an id that visit the site m times in the last nseconds,
given a list of logs with id and time sorted by time, returnall the bots's id
https://www.careercup.com/question?id=5756947954401280

建立一个class, 有get count 和 add event 两个methods. 就是计算给定时间内，有多少个event. 时间复杂度要求：至少O(logn).

Define a bot as an IP that hits the web app over M times in the past T seconds
 (not necessarily hits on the same page. Also take into account different API calls.) 
 How to design a bot detector layer and where to place it in the system.

就是有n个人, 比赛, 问你有多少种比赛结果排名,每个人可以独自一人一组,
也可以和其他人组成团体,
比如n= 2, 两个人A,B,
可能的结果有3种
A 第一,B 第二
B 第一,A 第二
A, B 团体第一
比赛排名，有并列（之前也有面经用dp）
http://www.1point3acres.com/bbs/thread-283244-1-1.html

单调栈计算容积 Trapping Rain Water

top k largest from an array, sort, heap, quick select 三种方法都让写一遍, 然后让证明quick select的平均复杂度是O(n)的
"""