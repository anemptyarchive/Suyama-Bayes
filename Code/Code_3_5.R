
# ch3.5 線形回帰の例 ------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvtnorm)

# ch3.5.1 モデルの構築 ----------------------------------------------------------

# ノイズなしの観測モデルの関数を定義
f_wx <- function(w_m, x) {
  y <- 0
  for(i in seq_along(w_m)) {
    tmp_term = w_m[i] * x^(i-1)
    y = y + tmp_term
  }
  y
}

# 観測モデルのパラメータを指定
M <- 4
sigma <- 0.33
lambda <- solve(sigma^2) %>% 
  as.numeric()

# 事前分布のパラメータを指定
m_m <- rep(0, M)
sigma_mm <- diag(M)
lambda_mm <- solve(sigma_mm^2)

# 観測モデルのパラメータをサンプリング
w_m <- mvtnorm::rmvnorm(n = 1, mean = m_m, sigma = sigma_mm^2) %>% 
  as.vector()


# 作図用のx軸の値
x_vec <- seq(-1, 1, by = 0.01)

# ノイズを含まないモデルを計算
line_df <- tibble(
  x = x_vec, 
  y = f_wx(w_m, x)
)

# 作図
ggplot(line_df, aes(x, y)) + 
  geom_line()


# 観測モデルによるデータをサンプリング
N <- 30
sample_df <- tibble(
  x = sample(seq(min(x_vec), max(x_vec), by = 0.01), size = N, replace = TRUE), # データをサンプリング
  eps = rnorm(n = N, mean = 0, sd = 1 / lambda), # 誤差項を生成
  y = f_wx(w_m, x) + eps
)

# 作図
ggplot() + 
  geom_point(data = sample_df, aes(x, y)) + 
  geom_line(data = line_df, aes(x, y), color = "blue")


# 観測モデルのパラメータのサンプリングによるモデルの変化を比較
smp_line_df <- tibble()
for(i in 1:5) {
  tmp_df <- tibble(
    x = x_vec, 
    y = f_wx(
      w_m = as.vector(mvtnorm::rmvnorm(n = 1, mean = m_m, sigma = solve(lambda_mm))), 
      x = x
    ), 
    smp_num = as.factor(i)
  )
  smp_line_df <- rbind(smp_line_df, tmp_df)
}
ggplot(smp_line_df, aes(x, y, color = smp_num)) + 
  geom_line()

