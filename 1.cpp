#include <iostream>
using namespace std;

int main(void){    

int [] prices = {7,1,5,3,6,4}
int[][] dp = new int[n][2];                                               /*定义main()函数*/
for (int i = 0; i < n; i++) {
    if (i - 1 == -1) {
        dp[i][0] = 0;
        dp[i][1] = -prices[i];
        continue;
    }
    dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i]);
    dp[i][1] = Math.max(dp[i-1][1], -prices[i]);
}
return dp[n - 1][0];
}
