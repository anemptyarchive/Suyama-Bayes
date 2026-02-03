
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.1
# 平均が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth <- 25

# 既知のパラメータを指定
lambda <- 0.01
1/sqrt(lambda) # 標準偏差


#### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_size <- (1/sqrt(lambda)) |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., abs(x_n-mu_truth))})() |> # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu_truth - x_size
x_max <- mu_truth + x_size

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


#### 分布の計算 -----

# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu_truth,       # 平均パラメータ
  lambda = lambda,         # 精度パラメータ
  sigma  = 1/sqrt(lambda), # 標準偏差パラメータ
  dens   = dnorm(x = x, mean = mu, sd = sigma) # 確率密度
)


### 事前分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
m         <- 0
lambda_mu <- 0.001


#### 変数の設定 -----

# μ軸の範囲を設定
mu_min <- x_min
mu_max <- x_max

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 1001)


#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_df <- tibble::tibble(
  mu        = mu_vec, # 確率変数
  m         = m,                 # 平均パラメータ
  lambda_mu = lambda_mu,         # 精度パラメータ
  sigma_mu  = 1/sqrt(lambda_mu), # 標準偏差パラメータ
  dens      = dnorm(x = mu, mean = m, sd = sigma_mu) # 確率密度
)


### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N <- 50

# 観測データを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda))


### データの集計 -----

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


### 事後分布(ガウス分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.53, 3.54)
lambda_mu_hat <- N * lambda + lambda_mu
m_hat         <- (sum(x_n) * lambda + m * lambda_mu) / lambda_mu_hat


#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_df <- tibble::tibble(
  mu        = mu_vec, # 確率変数
  m         = m_hat,             # 平均パラメータ
  lambda_mu = lambda_mu_hat,     # 精度パラメータ
  sigma_mu  = 1/sqrt(lambda_mu), # 標準偏差パラメータ
  dens      = dnorm(x = mu, mean = m, sd = sigma_mu) # 確率密度
)


#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ",               N, ", ", 
  "mu[truth] == ",       round(mu_truth,      digits = 2), ", ", 
  "hat(m) == ",          round(m_hat,         digits = 2), ", ", 
  "hat(lambda)[mu] == ", round(lambda_mu_hat, digits = 5), 
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
    color = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Gaussian distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(mu), 
    y = "density"
  )


### 予測分布(ガウス分布)を計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.62')
mu_star_hat     <- m_hat
lambda_star_hat <- lambda * lambda_mu_hat / (lambda + lambda_mu_hat)

mu_star_hat     <- (lambda * sum(x_n) + m * lambda_mu) / (N * lambda + lambda_mu)
lambda_star_hat <- (N * lambda + lambda_mu) * lambda / ((N+1) * lambda + lambda_mu)


#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_df <- tibble::tibble(
  x           = x_vec, # 確率変数
  mu_star     = mu_star_hat,         # 平均パラメータ
  lambda_star = lambda_star_hat,     # 精度パラメータ
  sigma_star  = 1/sqrt(lambda_star), # 標準偏差パラメータ
  dens        = dnorm(x = x, mean = mu_star, sd = sigma_star) # 確率密度
)


#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ",                N, ", ", 
  "mu[truth] == ",        round(mu_truth,        digits = 2), ", ", 
  "lambda == ",           round(lambda,          digits = 5), ", ", 
  "hat(mu)['*'] == ",     round(mu_star_hat,     digits = 2), ", ", 
  "hat(lambda)['*'] == ", round(lambda_star_hat, digits = 5), 
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
    title = "Gaussian distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "density"
  )


