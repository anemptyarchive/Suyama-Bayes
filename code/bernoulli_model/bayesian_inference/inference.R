
# ベルヌーイモデル -------------------------------------------------------------

# chapter 3.2.1
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ベルヌーイ分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth <- 0.25


#### 変数の設定 -----

# x軸の範囲を設定
x_min <- 0
x_max <- 1

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


#### 分布の計算 -----

# 生成分布の確率を計算
model_df <- tibble::tibble(
  x    = x_vec, # 確率変数
  mu   = mu_truth, # 成功確率パラメータ
  prob = c(1-mu_truth, mu_truth) # 確率
)


### 事前分布(ベータ分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


#### 変数の設定 -----

# μ軸の範囲を設定
mu_min <- 0
mu_max <- 1

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 1001)


#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_df <- tibble::tibble(
  mu   = mu_vec, # 確率変数
  a    = a, # パラメータ
  b    = b, # パラメータ
  dens = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
)


### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N <- 50

# 観測データを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)


#### データの集計 -----

# 観測データを集計
obs_df <- tibble::tibble(
  x        = x_vec,                   # 観測値
  freq     = c(N-sum(x_n), sum(x_n)), # 度数
  rel_freq = freq / N                 # 相対度数
)


### 事後分布(ベータ分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.15)
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b


#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_df <- tibble::tibble(
  mu   = mu_vec, # 確率変数
  a    = a_hat, # パラメータ
  b    = b_hat, # パラメータ
  dens = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
)


#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "mu[truth] == ", round(mu_truth, digits = 2), ", ", 
  "a == ", round(a, digits = 1), ", ", 
  "b == ", round(b, digits = 1), ", ", 
  "hat(a) == ", round(a_hat, digits = 1), ", ", 
  "hat(b) == ", round(b_hat, digits = 1), 
  ")"
) |> 
  parse(text = _)

# 事後分布を作図
ggplot() + 
  geom_vline(
    mapping = aes(xintercept = mu_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = prior_df, 
    mapping = aes(x = mu, y = dens, color = "prior"), 
    linewidth = 1, linetype = "dotted"
  ) + # 事前分布
  geom_line(
    data    = posterior_df, 
    mapping = aes(x = mu, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = mu_truth, 
      labels    = expression(mu[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "prior", "posterior"), 
    values = c("red", "purple", "purple"), 
    labels = c("true parameter", "prior distribution", "posterior distribution"), 
    name = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5)), 
  ) + 
  labs(
    title = "Beta distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(mu), 
    y = "density"
  )


### 予測分布(ベルヌーイ分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.19')
mu_star_hat <- a_hat / (a_hat + b_hat)
#mu_star_hat <- (sum(x_n) + a) / (N + a + b)


#### 分布の計算 -----

# 予測分布の確率を計算
predict_df <- tibble::tibble(
  x    = x_vec, # x軸の値
  mu   = mu_star_hat, # 成功確率パラメータ
  prob = c(1-mu_star_hat, mu_star_hat) # 確率
)


#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "mu[truth] == ", round(mu_truth, digits = 2), ", ", 
  "hat(mu)['*'] == ", round(mu_star_hat, digits = 1), 
  ")"
) |> 
  parse(text = _)

# 予測分布を作図
ggplot() + 
  geom_bar(
    data    = model_df, 
    mapping = aes(x = x, y = prob, color = "model"), 
    stat = "identity", position = "identity",
    fill = NA, linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_bar(
    data    = predict_df, 
    mapping = aes(x = x, y = prob, color = "predict"), 
    stat = "identity", position = "identity", 
    fill = "purple", alpha = 0.5, linetype = "blank"
  ) + # 予測分布
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  scale_color_manual(
    breaks = c("model", "predict"), 
    values = c("red", NA), 
    labels = c("true model", "predict distribution"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5)), 
  ) + 
  labs(
    title = "Bernoulli distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "probability"
  )


