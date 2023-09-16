import heapq


# Assisted by GPT
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        pq = []
        best = {}
        best[(0, 0)] = 0
        heapq.heappush(pq, ((0), (0, 0))) #((effort, len), (i, j))
        while True:
            (effort), (i, j) = heapq.heappop(pq)
            if i == len(heights) - 1 and j == len(heights[0]) - 1:
                return effort
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ni, nj = i + di, j + dj
                if 0 <= ni < len(heights) and 0 <= nj < len(heights[0]):
                    neffort = max(effort, abs(heights[i][j] - heights[ni][nj]))
                    if (ni, nj) not in best or best[(ni, nj)] > neffort:
                        best[(ni, nj)] = neffort
                        heapq.heappush(pq, ((neffort), (ni, nj)))


s = Solution()
s.minimumEffortPath([[1,2,2],[3,8,2],[5,3,5]])