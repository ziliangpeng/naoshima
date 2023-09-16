from typing import List

# Copilot.
class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        cliff = -1
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                cliff = i
                break
        if cliff == -1:
            return 0

        for i in range(cliff + 1, len(nums)-1):
            if nums[i] > nums[i+1]:
                return -1

        for i in range(cliff):
            if nums[i] > nums[i+1]:
                return -1

        if nums[-1] > nums[0]:
            return -1

        return len(nums) - 1 - cliff
        

s = Solution()
print(s.minimumRightShifts([3,4,5,1,2]))