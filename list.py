from collections import Counter

"""
1. Given an array and a target, return the number of pairs whose sum is the target.
10/12/2018 21:00

Idea: For special cases, list example and calculate the correct answer by math.

https://www.geeksforgeeks.org/count-pairs-with-given-sum/
"""

# [1, 2, 4, 5, 3, 1, 2, 0] t = 4 => (4, 0), (3, 1), (3, 1), (2, 2)

print("1. count_pair")

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

print("6. find_popular")


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

print("7. Iterative Merge Sort")

print()

"""
Merge 2 sorted array; Merge k Sorted Lists
给出一个sorted A, 和Sorted B, 并且B的长度是A的两倍，B组数列后半个组是空的。要求把这2个merge起来，sorted，并且不要extra space

给一个数, 如何判断是不是斐波那契数

给一个时间，比如9:30，返回时针和分针的夹角

K Inverse Pairs Array

Longest Substring Without Repeating Characters

给定一些字符串['hello', 'world', 'java'] 按要求输出'hwj', 'hwa', 'hwv', 'hwa', 'hoj' ... 
要求不用递归。follow up, 如何优化空间复杂度， follow up up, 如果每台机器内存只有所有输出的1 / 10，
如何利用hadoop平均分配。（题主不熟悉hadoop, 只说了一个分治的思想

Longest Consecutive Sequence

3Sum / Subarray Sum Equals K

LRU, 不让用dummy head tail
有两个follow up: 一个是在现实的project当中调用get（key），当key不存在的时候怎么处理；
另一个是，当cache的capacity满的时候，如何让其自动扩容

题目是给一些关系，比如 A 和 B的关系， B和 C的关系... 然后给一个 start 和target， print 出所有可以从start 出发，
在target截止，并且带上 relationship. 比如 A brother B, B mother C,  B mother A, B friend D.  给 start = A, 
target = C 的话， 要print 出 A brother B, B mother C。

设计一个21点的赌博游戏。 游戏里有1个庄家，k个玩家，玩法是 每个玩家刚开始有2张牌，面朝上，庄家2张牌，一张面朝上。
 从第一个玩家开始发牌，可以选择hit/pass. 如果hit的话就加分，但如果超过21点就输（可以选择很多次，如果玩家和庄家都没有超过21点， 
 比谁的分大。 最后print 出每个玩家赢还是庄家赢。主要考点是OOD，具体怎么实现不是特别care, 比如洗牌什么的只问了用什么data structure
  好之类的。

给出2个数组，比如A =[1,6,9], B = [1,1,1]. 每个数字相加，但是保证每个slot 的值小于10， 如果大于10 就要分开。
比如刚才2个A, B 的结果应该输出 [2,7,1,0];

给出一个10K的文档，里面很多文章。现在给出一个String query, 找到最有关系的10个文章。
有很多文章，实现一个很简单的搜索引擎，每篇文章对应一个word count，然后根据word去找

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

建立一个class, 有get count 和 add event 两个methods. 就是计算给定时间内，有多少个event. 时间复杂度要求：至少O(logn).

就是有n个人, 比赛, 问你有多少种比赛结果排名,每个人可以独自一人一组,
也可以和其他人组成团体,
比如n= 2, 两个人A,B,
可能的结果有3种
A 第一,B 第二
B 第一,A 第二
A, B 团体第一
比赛排名，有并列（之前也有面经用dp）
http://www.1point3acres.com/bbs/thread-283244-1-1.html

森林里面一群兔子，你去问了一些兔子有多少只兔子跟你一个颜色，得到一个数组，问森林里面至少有多少只兔子
例子 [1,2,1] 有两只兔子说有一只跟他一个颜色，这两只可能是一对同色的兔子，还有一只兔子说有两只跟自己一个颜色，所以这个颜色的兔子至少有三只，加起来就是最少有五只兔子
小兔子脑筋急转弯，一个小兔子可以说我看到了几只和自己一样颜色的小兔子，问森林里最多有几只小兔子
（what!!!??? 比如2,2,3，意思是，第一只小兔子说有两只和我一样，第二只说有两只和我一样，第三只说有三只和我一样
，前面说话的两只可能是一种颜色的，说3的那个不可能和前面一种颜色，所以一共五只小兔子，哈哈哈哈哈哈

单调栈计算容积 Trapping Rain Water   	

Number of Islands

top k largest from an array, sort, heap, quick select 三种方法都让写一遍, 然后让证明quick select的平均复杂度是O(n)的
"""