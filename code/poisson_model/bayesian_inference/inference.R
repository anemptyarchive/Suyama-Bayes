
# ポアソンモデル ---------------------------------------------------------------

# chapter 3.2.3
# ベイズ推論
# 推論アルゴリズムの実装


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
lambda_truth <- 4


#### 変数の設定 -----

# x軸の範囲を設定
u <- 5
x_min <- 0
x_max <- lambda_truth |> # 基準値を指定
  (\(.) {. * 3})() |> # 倍率を指定
  #(\(.) {max(., x_n)})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min; x_max

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


#### 分布の計算 -----

# 生成分布の確率を計算:式(2.37)
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
u <- 5
lambda_min <- 0
lambda_max <- lambda_truth |> # 基準値を指定
  (\(.) {. * 3})() |> # 倍率を指定
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 1001)


#### 分布の計算 -----

# 事前分布の確率密度を計算:式(2.56)
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

# ポアソンモデルのデータを生成
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

# 事後分布の確率密度を計算:式(2.56)
posterior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  a      = a_hat, # 形状パラメータ
  b      = b_hat, # 尺度パラメータ
  dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)


#### 分布の作図 -----

# ラベル用の文字列を作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
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
      breaks = lambda_truth, 
      labels = expression(lambda[truth])
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
    title = "Gamma distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(lambda), 
    y = "density"
  )


### 予測分布(負の二項分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat <- a_hat
p_hat <- 1 / (b_hat + 1)
#r_hat <- sum(x_n) + a
#p_hat <- 1 / (N + b + 1)


#### 分布の計算 -----

# 予測分布の確率を計算:式(3.43)
predict_df <- tibble::tibble(
  x    = x_vec, # 確率変数
  r    = r_hat, # 成功回数パラメータ
  p    = p_hat, # 失敗確率パラメータ
  prob = dnbinom(x = x, size = r_hat, prob = 1-p_hat) # 確率
)


#### 分布の作図 -----

# ラベル用の文字列を作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
  "hat(r) == ", round(r_hat, digits = 1), ", ", 
  "hat(p) == ", round(p_hat, digits = 5), 
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
  scale_x_continuous(
    breaks = x_vec, minor_breaks = FALSE, 
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks = lambda_truth, 
      labels = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
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
    title = "Negative Binomial distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "probability"
  )



# ポアソンモデル ---------------------------------------------------------------

# chapter 3.2.3
# ベイズ推論
# 学習推移の可視化


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth <- 4


### 観測データの設定 -----

# (ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# ポアソンモデルのデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)


### 変数の設定 -----

# x軸の範囲を設定
u <- 5
x_min <- 0
x_max <- lambda_truth |> # 基準値を指定
  (\(.) {. * 3})() |> # 倍率を指定
  (\(.) {max(., x_n)})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
cat('x size:', x_min, x_max)

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


# λ軸の範囲を設定
lambda_min <- x_min
lambda_max <- x_max
cat('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 1001)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
a <- 1
b <- 1


# 予測分布のパラメータを計算:式(3.44)
r <- a
p <- 1 / (b + 1)

# 受け皿を初期化
trace_a_i <- rep(NA, times = N+1)
trace_b_i <- rep(NA, times = N+1)
trace_r_i <- rep(NA, times = N+1)
trace_p_i <- rep(NA, times = N+1)

# 初期値を記録
trace_a_i[1] <- a
trace_b_i[1] <- b
trace_r_i[1] <- r
trace_p_i[1] <- p

# パラメータを更新
for(n in 1:N) {
  
  # 観測データを取得
  x <- x_n[n]
  
  # 事後分布のパラメータを更新:式(3.38)
  a <- a + x
  b <- b + 1
  
  # 予測分布のパラメータを更新:式(3.44)
  r <- a
  p <- 1 / (b + 1)
  #r <- r + x
  #p <- 1 / (1/p + 1)
  
  # 更新値を記録
  trace_a_i[n+1] <- a
  trace_b_i[n+1] <- b
  trace_r_i[n+1] <- r
  trace_p_i[n+1] <- p
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
a <- 1
b <- 1


# 事後分布のパラメータを計算:式(3.38)
trace_a_i <- c(a, cumsum(x_n) + a)
trace_b_i <- 0:N + b

# 予測分布のパラメータを計算:式(3.44')
trace_r_i <- c(a, cumsum(x_n) + a)
trace_p_i <- 1 / (0:N + b + 1)


### 推移の作図 -----

# 観測データを格納
anim_obs_df <- tibble::tibble(
  n = 0:N,       # データ番号
  x = c(NA, x_n) # 観測値
)


#### 事後分布の作図 -----

# 事後分布の確率を計算:式(3.38)
anim_posterior_df <- tidyr::expand_grid(
  n      = 0:N,       # 試行回数
  lambda = lambda_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    a    = trace_a_i[n+1], # 形状パラメータ
    b    = trace_b_i[n+1], # 尺度パラメータ
    dens = dgamma(x = lambda, shape = a, rate = b) # 確率密度
  )

# ラベル用の文字列を作成
anim_param_df <- tibble::tibble(
  n = 0:N, 
  a = trace_a_i, 
  b = trace_b_i, 
  posterior_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
    "hat(a) == ", round(a, digits = 1), ", ", 
    "hat(b) == ", round(b, digits = 1), 
    ")"
  )
)


# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_vline(
    mapping = aes(xintercept = lambda_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = anim_posterior_df, 
    mapping = aes(x = lambda, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  geom_point(
    data    = anim_obs_df, 
    mapping = aes(x = x, y = 0, color = "data"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = posterior_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -1
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks = lambda_truth, 
      labels = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "posterior", "data"), 
    values = c("red", "purple", "pink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        linewidth = c(0.5, 0.5, NA), 
        linetype  = c("dashed", "solid", NA), 
        size      = c(NA, NA, 5), 
        shape     = c(NA, NA, "circle"))
    )
  ) + 
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Gamma distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = expression(lambda), 
    y = "density"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/poisson/parameter_updates/posterior.mp4")
)


#### 予測分布の作図 -----

# 生成分布の確率を計算:式(2.37)
anim_model_df <- tidyr::expand_grid(
  n = 0:N, # データ番号
  tibble::tibble(
    x      = x_vec, # 確率変数
    lambda = lambda_truth, # 期待値パラメータ
    prob   = dpois(x = x, lambda = lambda) # 確率
  )
) # 試行ごとに分布を複製

# 予測分布の確率を計算:式(3.44)
anim_predict_df <- tidyr::expand_grid(
  n = 0:N,  # 試行回数
  x = x_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    r    = trace_r_i[n+1], # 成功回数パラメータ
    p    = trace_p_i[n+1], # 失敗確率パラメータ
    prob = dnbinom(x = x, size = r, prob = 1-p), # 確率
  )

# ラベル用の文字列を作成
anim_param_df <- tibble::tibble(
  n = 0:N, 
  r = trace_r_i, 
  p = trace_p_i, 
  predict_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
    "hat(r) == ", round(r, digits = 1), ", ", 
    "hat(p) == ", round(p, digits = 5), 
    ")"
  )
)


# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_bar(
    data    = anim_model_df, 
    mapping = aes(x = x, y = prob, color = "model"), 
    stat = "identity", position = "identity",
    fill = NA, linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_bar(
    data    = anim_predict_df, 
    mapping = aes(x = x, y = prob, color = "predict"), 
    stat = "identity", position = "identity", 
    fill = "purple", alpha = 0.5, linetype = "blank"
  ) + # 予測分布
  geom_point(
    data    = anim_obs_df, 
    mapping = aes(x = x, y = 0, color = "data"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = predict_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -1
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_x_continuous(
    breaks = x_vec, minor_breaks = FALSE, 
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks = lambda_truth, 
      labels = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "predict", "data"), 
    values = c("red", NA, "pink"), 
    labels = c("true model", "predict distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        size      = c(NA, NA, 5), 
        shape     = c(NA, NA, "circle"), 
        linewidth = c(0.5, 0, NA), 
        linetype  = c("dashed", "blank", NA)
      )
    )
  ) + 
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Negative Binomial distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = expression(x), 
    y = "probability"
  )

# 動画を作成
gganimate::animate(
  plot = predict_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/poisson/parameter_updates/predict.mp4")
)


