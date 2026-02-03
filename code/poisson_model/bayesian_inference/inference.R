
# ポアソンモデル ---------------------------------------------------------------

# chapter 3.2.3
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
lambda_truth <- 4


#### 変数の設定 -----

# x軸の範囲を設定
k <- 3
u <- 5
x_min <- 0
x_max <- lambda_truth |> # 期待値
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., x_n)})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


#### 分布の計算 -----

# 生成分布の確率を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  lambda = lambda_truth, # 期待値パラメータ
  prob   = dpois(x = x, lambda = lambda) # 確率
)


### 事前分布(ガンマ分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


#### 変数の設定 -----

# λ軸の範囲を設定
k <- 3
u <- 5
lambda_min <- 0
lambda_max <- lambda_truth |> # 真のパラメータ
  (\(.) {. * k})() |> # 定数倍
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 1001)


#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  a      = a, # 形状パラメータ
  b      = b, # 尺度パラメータ
  dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)


### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N <- 50

# 観測データを生成
x_n <- rpois(n = N ,lambda = lambda_truth)


#### データの集計 -----

# 観測データを集計
obs_df <- tidyr::tibble(
  x = x_n # 観測値
) |> 
  dplyr::count(
    x, name = "freq" # 度数
  ) |> 
  dplyr::mutate(
    rel_freq = freq / N # 相対度数
  ) |> 
  tidyr::complete(
    x = x_vec, 
    fill = list(freq = 0, rel_freq = 0)
  ) # 未観測値を補完


### 事後分布(ガンマ分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.38)
a_hat <- sum(x_n) + a
b_hat <- N + b


#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  a      = a_hat, # 形状パラメータ
  b      = b_hat, # 尺度パラメータ
  dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)


#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ",             N, ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
  "a == ",             round(a,            digits = 1), ", ", 
  "b == ",             round(b,            digits = 1), ", ", 
  "hat(a) == ",        round(a_hat,        digits = 1), ", ", 
  "hat(b) == ",        round(b_hat,        digits = 1), 
  ")"
) |> 
  parse(text = _)

# 事後分布を作図
ggplot() + 
  geom_vline(
    mapping = aes(xintercept = lambda_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = prior_df, 
    mapping = aes(x = lambda, y = dens, color = "prior"), 
    linewidth = 1, linetype = "dotted"
  ) + # 事前分布
  geom_line(
    data    = posterior_df, 
    mapping = aes(x = lambda, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = lambda_truth, 
      labels    = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "prior", "posterior"), 
    values = c("red", "purple", "purple"), 
    labels = c("true parameter", "prior distribution", "posterior distribution"), 
    name = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Gamma distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(lambda), 
    y = "density"
  )


### 予測分布(負の二項分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat <- a_hat
q_hat <- 1 / (1 + b_hat)
p_hat <- b_hat / (1 + b_hat)

r_hat <- sum(x_n) + a
q_hat <- 1 / (1 + N + b)
p_hat <- (N + b) / (1 + N + b)

p_hat <- 1 - q_hat


#### 分布の計算 -----

# 予測分布の確率を計算
predict_df <- tibble::tibble(
  x    = x_vec, # 確率変数
  r    = r_hat, # 成功回数パラメータ
  p    = p_hat, # 成功確率パラメータ
  prob = dnbinom(x = x, size = r, prob = p) # 確率
)


#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ",             N, ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
  "hat(r) == ",        round(r_hat,        digits = 1), ", ", 
  "hat(p) == ",        round(p_hat,        digits = 5), 
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
    color = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Negative Binomial distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "probability"
  )


