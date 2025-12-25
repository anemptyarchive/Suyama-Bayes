
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.2
# 精度が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)
library(LaplacesDemon)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 既知のパラメータを指定
mu <- 5

# 真のパラメータを指定
lambda_truth <- 0.25
1/sqrt(lambda_truth)


#### 変数の設定 -----

# x軸の範囲を設定
u <- 5
k <- 4
x_size <- (1/sqrt(lambda_truth)) |> # 基準値を指定
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., abs(x_n-mu))})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu - x_size
x_max <- mu + x_size

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, length.out = 1001)


#### 分布の計算 -----

# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu,           # 平均パラメータ
  lambda = lambda_truth, # 精度パラメータ
  sigma  = 1/sqrt(lambda), # 標準偏差パラメータ
  dens   = dnorm(x = x, mean = mu, sd = sigma) # 確率密度
)


### 事前分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


#### 変数の設定 -----

# λ軸の範囲を設定
u <- 0.5
k <- 3
lambda_min <- 0
lambda_max <- lambda_truth |> # 基準値を指定
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
x_n <- rnorm(n = N, mean = mu, sd = 1/sqrt(lambda_truth))


#### データの集計 -----

# 階級数を指定
bin_num <- 40

# 階級幅を計算
bin_size <- (x_max - x_min) / bin_num

# 境界値の範囲を設定
bin_min <- x_min - 0.5*bin_size
bin_max <- x_max + 0.5*bin_size

# 観測データを集計
obs_df <- tibble::tibble(
  x = x_n # サンプル値
) |> 
  dplyr::mutate(
    bin_i  = (x - bin_min) %/% bin_size,        # 階級番号
    center = bin_min + (bin_i + 0.5) * bin_size # 階級値
  ) |> 
  dplyr::count(
    center, name = "freq" # 度数
  ) |> 
  dplyr::mutate(
    dens = freq / (bin_size * N) # 密度
  ) |> 
  tidyr::complete(
    center = seq(from = x_min, to = x_max, by = bin_size), 
    fill = list(freq = 0, dens = 0)
  ) # 未観測値を補完


### 事後分布(ガンマ分布)の計算 -----

#### パラメータの計算 -----

# λの事後分布のパラメータを計算:式(3.69)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * sum((x_n - mu)^2) + b


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
  "N == ", N, ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 5), ", ", 
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
    linetype = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Gamma distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(lambda), 
    y = "density"
  )


### 予測分布(スチューデントのt分布)を計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.79)
mu_s         <- mu
lambda_s_hat <- a_hat / b_hat
nu_s_hat     <- 2 * a_hat

lambda_s_hat <- (N + 2 * a) / (sum((x_n - mu)^2) + 2 * b)
nu_s_hat     <- N + 2 * a


#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_df <- tibble::tibble(
  x        = x_vec, # 確率変数
  mu_s     = mu_s,             # 位置パラメータ
  lambda_s = lambda_s_hat,     # 逆尺度パラメータ
  sigma_s  = 1/sqrt(lambda_s), # 尺度パラメータ
  nu_s     = nu_s_hat,         # 自由度パラメータ
  dens     = LaplacesDemon::dst(x = x, mu = mu_s, sigma = sigma_s, nu = nu_s) # 確率密度
)


#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ",              N, ", ", 
  "mu == ",             round(mu,           digits = 2), ", ", 
  "lambda[truth] == ",  round(lambda_truth, digits = 5), ", ", 
  "mu[s] == ",          round(mu_s,         digits = 2), ", ", 
  "hat(lambda)[s] == ", round(lambda_s_hat, digits = 5), ", ", 
  "hat(nu)[s] == ",     round(nu_s_hat,     digits = 1), 
  ")"
) |> 
  parse(text = _)

# 予測分布を作図
ggplot() + 
  geom_line(
    data    = model_df, 
    mapping = aes(x = x, y = dens, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_line(
    data    = predict_df, 
    mapping = aes(x = x, y = dens, color = "predict"), 
    linewidth = 1
  ) + # 予測分布
  scale_color_manual(
    breaks = c("model", "predict"), 
    values = c("red", "purple"), 
    labels = c("true model", "predict distribution"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Student's t distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "density"
  )


