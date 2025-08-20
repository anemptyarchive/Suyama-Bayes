
# ポアソンモデル ---------------------------------------------------------------

# chapter 3.2.3
# ベイズ推論の実装
# 学習推移の可視化


# ライブラリの読込 ----------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth <- 4


### 観測データの生成 -----

# データ数を指定
N <- 10

# (ノートとの対応用)
set.seed(86)

# ポアソンモデルのデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)


### 変数の設定 -----

# x軸の範囲を設定
u <- 5
x_max <- lambda_truth |> # 基準値を指定
  (\(.) {. * 3})() |> # 倍率を指定
  (\(.) {max(., x_n)})() |> # # 乱数と比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# x軸の値を作成
x_vec <- seq(from = 0, to = x_max, by = 1)


# 真の分布を計算:式(2.37)
model_df <- tibble::tibble(
  x     = x_vec, # 確率変数
  lambd = lambda_truth, # 期待値パラメータ
  prob  = dpois(x = x, lambda = lambda) # 確率
)

# 観測データを集計
obs_df <- tidyr::tibble(
  x = x_n # 観測値
) |> 
  dplyr::count(
    x, name = "freq" # 度数
  ) |> 
  tidyr::complete(
    x = x_vec, 
    fill = list(freq = 0)
  ) # 未観測値を補完


### 事前分布(ガンマ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


### 変数の設定 -----

# λ軸の範囲を設定
u <- 5
lambda_max <- lambda_truth |> # 基準値を指定
  max() |> 
  (\(.) {. * 3})() |> # 倍率を指定
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# λ軸の値を作成
lambda_vec <- seq(from = 0, to = lambda_max, length.out = 1001)


# 事前分布を計算:式(2.56)
prior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  a      = a, # 形状パラメータ
  b      = b, # 尺度パラメータ
  dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)


### 事後分布(ガンマ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.38)
a_hat <- sum(x_n) + a
b_hat <- N + b


# 事後分布を計算:式(2.56)
posterior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  a      = a_hat, # 形状パラメータ
  b      = b_hat, # 尺度パラメータ
  dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)


# ラベル用の文字列を作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "hat(a) == ", a_hat, ", ", 
  "hat(b) == ", b_hat, 
  ")"
) |> 
  parse(text = _)

# 事後分布を作図
ggplot() + 
  geom_line(
    data    = posterior_df, 
    mapping = aes(x = lambda, y = dens, linetype = "posterior"), 
    color = "purple", linewidth = 1
  ) + # 事後分布
  geom_vline(
    mapping = aes(xintercept = lambda_truth, linetype = "param"), 
    color = "red", linewidth = 1
  ) + # 真のパラメータ
  scale_linetype_manual(
    breaks = c("param", "posterior"), 
    values = c("dashed", "solid"), 
    labels = c("true parameter", "posterior distribution"), 
    name = ""
  ) + # (凡例表示用)
  guides(
    linetype = guide_legend(override.aes = list(linewidth = 0.5)), 
  ) + # 凡例の体裁
  labs(
    title = "Gamma distribution", 
    subtitle = posterior_param_lbl, 
    x = expression(lambda), 
    y = "density"
  )


### 予測分布(負の二項分布)の計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat <- a_hat
p_hat <- 1 / (b_hat + 1)
r_hat <- sum(x_n) + a
p_hat <- 1 / (N + b + 1)


# 予測分布を計算:式(3.43)
predict_df <- tibble::tibble(
  x    = x_vec, # 確率変数
  r    = r_hat, # 成功回数の閾値パラメータ
  p    = p_hat, # 失敗確率パラメータ
  prob = dnbinom(x = x, size = r_hat, prob = 1-p_hat) # 確率
)


# ラベル用の文字列を作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ", N, ", ", 
  "hat(r) == ", r_hat, ", ", 
  "hat(p) == ", round(p_hat, digits = 5), 
  ")"
) |> 
  parse(text = _)

# 予測分布を作図
ggplot() + 
  geom_bar(
    data    = predict_df, 
    mapping = aes(x = x, y = prob, linetype = "predict"), 
    stat = "identity", position = "identity", 
    fill = "purple", color = NA
  ) + # 予測分布
  geom_bar(
    data    = model_df, 
    mapping = aes(x = x, y = prob, linetype = "model"), 
    stat = "identity", position = "identity",
    fill = NA, color = "red", linewidth = 1
  ) + # 真の分布
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  scale_linetype_manual(
    breaks = c("model", "predict"), 
    values = c("dashed", "blank"), 
    labels = c("true model", "predict distribution"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    linetype = guide_legend(override.aes = list(linewidth = 0.5)), 
  ) + # 凡例の体裁
  labs(
    title = "Negative Binomial distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "probability"
  )


# 学習推移の可視化 -------------------------------------------------------------

### 観測データの設定 -----

# データ数(試行回数)を指定
N <- 100


### 生成分布(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth <- 4


### 推論処理 -----

#### 1データずつ更新 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1

# 初期値による予測分布のパラメータを計算:式(3.44)
r <- a
p <- 1 / (b + 1)

# 受け皿を作成
x_n       <- rep(NA, times = N)
trace_a_i <- rep(NA, times = N+1)
trace_b_i <- rep(NA, times = N+1)
trace_r_i <- rep(NA, times = N+1)
trace_p_i <- rep(NA, times = N+1)

# 初期値を格納
trace_a_i[1] <- a
trace_b_i[1] <- b
trace_r_i[1] <- r
trace_p_i[1] <- p

# ベイズ推論による推定
for(n in 1:N) {
  
  # ポアソンモデルのデータを生成
  x_n[n] <- rpois(n = 1 ,lambda = lambda_truth)
  
  # 事後分布のパラメータを更新:式(3.38)
  a <- a + x_n[n]
  b <- b + 1
  
  # 予測分布のパラメータを更新:式(3.44)
  r <- a
  p <- 1 / (b + 1)
  #r <- r + x_n[n]
  #p <- 1 / (1/p + 1)
  
  # 更新値を格納
  trace_a_i[n+1] <- a
  trace_b_i[n+1] <- b
  trace_r_i[n+1] <- r
  trace_p_i[n+1] <- p
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### tidyverseパッケージによる処理 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1

# ポアソンモデルのデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)

# 試行ごとに事後分布を計算:式(3.38)
anim_posterior_df <- tidyr::expand_grid(
  n      = 0:N, # 試行回数
  lambda = lambda_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    a    = c(a, cumsum(x_n) + a)[n+1], 
    b    = n + b, 
    #a    = trace_a_i[n+1], # (「1データずつ更新」用)
    #b    = trace_b_i[n+1], # (「1データずつ更新」用)
    dens = dgamma(x = lambda, shape = a, rate = b) # 確率密度
  )

# 試行ごとに予測分布を計算:式(3.44)
anim_predict_df <- tidyr::expand_grid(
  n = 0:N,  # 試行回数
  x = x_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    r    = c(a, cumsum(x_n) + a)[n+1], # 成功回数の閾値パラメータ
    p    = 1 / (n + b + 1),            # 失敗確率パラメータ
    #r    = trace_r_i[n+1], # (「1データずつ更新」用)
    #p    = trace_p_i[n+1], # (「1データずつ更新」用)
    prob = dnbinom(x = x, size = r, prob = 1-p), # 確率
  )


### 推移の作図 -----

# 観測データを格納
anim_data_df <- tibble::tibble(
  n = 0:N,       # 試行回数
  x = c(NA, x_n) # 観測値
)

# 更新値を取得
trace_a_i <- anim_posterior_df |> 
  dplyr::arrange(n) |> 
  dplyr::distinct(n, a) |> 
  dplyr::pull(a)
trace_b_i <- anim_posterior_df |> 
  dplyr::arrange(n) |> 
  dplyr::distinct(n, b) |> 
  dplyr::pull(b)
trace_r_i <- anim_predict_df |> 
  dplyr::arrange(n) |> 
  dplyr::distinct(n, r) |> 
  dplyr::pull(r)
trace_p_i <- anim_predict_df |> 
  dplyr::arrange(n) |> 
  dplyr::distinct(n, p) |> 
  dplyr::pull(p)

# ラベル用の文字列を作成
anim_param_df <- tibble::tibble(
  n = 0:N, # 試行回数
  a = trace_a_i, 
  b = trace_b_i, 
  r = trace_r_i, 
  p = trace_p_i, 
  posterior_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "hat(a) == ", a, ", ", 
    "hat(b) == ", b, 
    ")"
  ), # 事後分布のパラメータラベル
  predict_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "hat(r) == ", r, ", ", 
    "hat(p) == ", round(p, digits = 5), 
    ")"
  ) # 予測分布のパラメータラベル
)


# 事後分布のアニメーションを作図
posterior_graph <- ggplot() + 
  geom_line(
    data    = anim_posterior_df, 
    mapping = aes(x = lambda, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  geom_vline(
    mapping = aes(xintercept = lambda_truth, color = "param"), 
    size = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x, y = 0, color = "data"), na.rm = TRUE, 
    size = 6
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = posterior_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -0.5
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_color_manual(
    breaks = c("param", "posterior", "data"), 
    values = c("red", "purple", "pink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        size      = c(NA, NA, 5), 
        linewidth = c(0.5, 0.5, NA), 
        linetype  = c("dashed", "solid", NA), 
        shape     = c(NA, NA, "circle"))
      )
  ) + # 凡例の体裁
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + # 図の体裁
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Gamma distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "λ", 
    y = "density"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 12, height = 8, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/ch3_2_3/posterior.mp4")
)


# 真の分布を計算:式(2.37)
anim_model_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  tibble::tibble(
    x      = x_vec, # 確率変数
    lambda = lambda_truth, # 期待値パラメータ
    prob   = dpois(x = x, lambda = lambda) # 確率
  )
) # 試行ごとに分布を複製

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_bar(
    data    = anim_predict_df, 
    mapping = aes(x = x, y = prob, color = "predict"), 
    stat = "identity", position = "identity", 
    fill = "purple"
  ) + # 予測分布
  geom_bar(
    data    = anim_model_df, 
    mapping = aes(x = x, y = prob, color = "model"), 
    stat = "identity", position = "identity",
    fill = NA, linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x, y = 0, color = "data"), na.rm = TRUE, 
    size = 6
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = predict_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -0.5
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
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
        shape     = c(NA, NA, "circle")), 
        linewidth = c(0.5, 0.5, NA), 
        linetype  = c("dashed", "solid", NA)
    )
  ) + # 凡例の体裁
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + # 図の体裁
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Negative Binomial distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "x", 
    y = "probability"
  )

# 動画を作成
gganimate::animate(
  plot = predict_graph, 
  nframes = N+1, fps = 10, 
  width = 12, height = 8, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/ch3_2_3/predict.mp4")
)


