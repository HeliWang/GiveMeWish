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