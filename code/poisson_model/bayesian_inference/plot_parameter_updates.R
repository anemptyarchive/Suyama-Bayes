
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
q <- 1 / (1 + b)
p <- 1 - q

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
  q <- 1 / (1 + b)
  p <- b / (1 + b)
  #r <- r + x
  #q <- 1 / (1 + 1/q)
  #p <- 1 - q
  
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
trace_r_i <- trace_a_i
trace_q_i <- 1 / (1 + 0:N + b)
trace_p_i <- (0:N + b) / (1 + 0:N + b)


### 推移の作図 -----

# 観測データを格納
anim_obs_df <- tibble::tibble(
  n = 0:N,       # データ番号
  x = c(NA, x_n) # 観測値
)


#### 事後分布の作図 -----

# 事後分布の確率密度を計算
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
    data    = anim_sample_df, 
    mapping = aes(x = x, y = 0, color = "sample"), 
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
      breaks    = lambda_truth, 
      labels    = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", "purple", "hotpink"), 
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
    p    = trace_p_i[n+1], # 成功確率パラメータ
    prob = dnbinom(x = x, size = r, prob = p), # 確率
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


# 予測分布を作図
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
    data    = anim_sample_df, 
    mapping = aes(x = x, y = 0, color = "sample"), 
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
      breaks    = lambda_truth, 
      labels    = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "predict", "sample"), 
    values = c("red", NA, "hotpink"), 
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


#### 観測データと分布の関係 -----

# 一時書き出し先を指定
dir_path <- "figure/tmp_folder"


# 確率軸の範囲を設定
u <- 0.05
prob_max <- dnbinom(
  x    = ifelse(
    test = trace_r_i > 0, 
    yes  = floor((trace_r_i - 1) * (1 - trace_p_i) / trace_p_i), 
    no   = 0
  ), # 最頻値
  size = trace_r_i, 
  prob = trace_p_i
) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# 確率密度軸の範囲を設定
u <- 0.05
dens_max <- dgamma(
  x     = (trace_a_i - 1) / trace_b_i, # 最頻値
  shape = trace_a_i, 
  rate  = trace_b_i
) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

for(i in 1:(N+1)) {
  
  # 値を取得
  n <- i - 1 # データ番号
  x <- x_n[n] # 観測値
  a <- trace_a_i[i] # 形状パラメータ
  b <- trace_b_i[i] # 尺度パラメータ
  r <- trace_r_i[i] # 成功回数パラメータ
  p <- trace_p_i[i] # 成功確率パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x = x # 観測値
  )
  
  
  ##### 観測データの作図 -----
  
  # 生成分布の期待値を計算
  E_x <- lambda_truth
  
  # 観測データの標本平均を計算
  bar_x = mean(x_n[0:n])
  
  # 生成分布の確率を計算:式(2.37)
  model_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    prob = dpois(x = x, lambda = lambda_truth) # 確率
  )
  
  # 観測データを集計
  obs_df <- tibble::tibble(
    x = x_n[0:n] # 観測値
  ) |> 
    dplyr::count(
      x, name = "freq" # 度数
    ) |> 
    dplyr::mutate(
      rel_freq = freq / n # 相対度数
    ) |> 
    tidyr::complete(
      x = x_vec, 
      fill = list(freq = 0, rel_freq = 0)
    ) # 未観測値を補完
  
  
  # ラベル用の文字列を作成
  model_param_lbl <- paste0(
    "list(", 
    "N == ", n, ", ", 
    "lambda[truth] == ", round(lambda_truth, digits = 2), ", ", 
    "paste(E(x) == lambda[truth], {} == ", round(E_x, digits = 2), ")", ", ", 
    "bar(x) == ", round(bar_x, digits = 2), 
    ")"
  ) |> 
    parse(text = _)
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = lambda_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = bar_x), 
      color = "hotpink", linewidth = 1, linetype = "dashed"
    ) + # 標本平均
    geom_bar(
      data    = model_df, 
      mapping = aes(x = x, y = prob, color = "model"), 
      stat = "identity", position = "identity",
      fill = NA, linewidth = 1, linetype = "dashed"
    ) + # 真の分布
    geom_bar(
      data    = obs_df, 
      mapping = aes(x = x, y = rel_freq, color = "sample"), 
      stat = "identity", position = "identity", 
      fill = "hotpink", alpha = 0.5, linetype = "blank"
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      breaks = x_vec, minor_breaks = FALSE, 
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(E_x, bar_x), 
        labels    = c(expression(E(x)), expression(bar(x)))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      sec.axis = sec_axis(transform = ~ .*N, name = "frequency")
    ) + # 度数軸目盛
    scale_color_manual(
      breaks = c("model", "sample"), 
      values = c("red", NA), 
      labels = c("true model", "observation data"), 
      name   = ""
    ) + # (凡例表示用)
    guides(
      color = guide_legend(
        override.aes = list(
          linewidth = c(0.5, 0), 
          linetype  = c("dashed", "blank")
        )
      )
    ) + 
    coord_cartesian(
      xlim = c(x_min, x_max), # (目盛の共通化用)
      ylim = c(0, prob_max)
    ) + 
    labs(
      title = "Poisson distribution", 
      subtitle = model_param_lbl, 
      x = expression(x), 
      y = "probability"
    )
  model_graph
  
  
  ##### 事後分布の作図 -----
  
  # 事後分布の期待値を計算
  E_lambda <- a / b
  
  # 事後分布の確率を計算:式(3.38)
  posterior_df <- tibble::tibble(
    lambda = lambda_vec, # 確率変数
    dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
  )
  
  
  # ラベル用の文字列を作成
  posterior_param_lbl <- paste0(
    "list(", 
    "hat(a) == ", round(a, digits = 1), ", ", 
    "hat(b) == ", round(b, digits = 1), ", ", 
    "paste(E(lambda) == frac(a, b), {} == ", round(E_lambda, digits = 2), ")", 
    ")"
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = lambda_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_lambda), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_line(
      data    = posterior_df, 
      mapping = aes(x = lambda, y = dens, color = "posterior"), 
      linewidth = 1
    ) + # 事後分布
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(lambda_truth, E_lambda), 
        labels    = c(expression(lambda[truth]), expression(E(lambda)))
      ) # パラメータラベル
    ) + 
    scale_color_manual(
      breaks = "posterior", 
      values = "purple", 
      labels = "posterior distribution", 
      name   = ""
    ) + # (凡例表示用)
    guides(
      color = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(lambda_min, lambda_max), # (目盛の共通化用)
      ylim = c(0, dens_max)
    ) + 
    labs(
      title = "Gamma distribution", 
      subtitle = posterior_param_lbl, 
      x = expression(lambda), 
      y = "density"
    )
  posterior_graph
  
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x <- r * (1 - p) / p
  
  # 予測分布の確率を計算:式(3.44)
  predict_df <- tibble::tibble(
    x = x_vec, # 確率変数
    prob = dnbinom(x = x, size = r, prob = p), # 確率
  )
  
  # ラベル用の文字列を作成
  predict_param_lbl <- paste0(
    "list(", 
    "hat(r) == ", round(r, digits = 1), ", ", 
    "hat(p) == ", round(p, digits = 5), ", ", 
    "paste(E(x) == frac(r * (1-p), p), {} == ", round(E_x, digits = 2), ")", 
    ")"
  ) |> 
    parse(text = _)
  
  # 予測分布のアニメーションを作図
  predict_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = lambda_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_x), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_bar(
      data    = model_df, 
      mapping = aes(x = x, y = prob), 
      stat = "identity", position = "identity",
      fill = NA, color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真の分布
    geom_bar(
      data    = predict_df, 
      mapping = aes(x = x, y = prob, color = "predict"), 
      stat = "identity", position = "identity", 
      fill = "purple", alpha = 0.5, linetype = "blank"
    ) + # 予測分布
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      breaks = x_vec, minor_breaks = FALSE, 
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(lambda_truth, E_x), 
        labels    = c(expression(lambda[truth]), expression(E(x)))
      ) # パラメータラベル
    ) + 
    scale_color_manual(
      breaks = "predict", 
      values = NA, 
      labels = "predict distribution", 
      name   = ""
    ) + # (凡例表示用)
    guides(
      color = guide_legend(override.aes = list(linetype  = "blank"))
    ) + 
    coord_cartesian(
      xlim = c(x_min, x_max), # (目盛の共通化用)
      ylim = c(0, prob_max)
    ) + 
    labs(
      title = "Negative Binomial distribution", 
      subtitle = predict_param_lbl, 
      x = expression(x), 
      y = "probability"
    )
  predict_graph
  
  
  ##### グラフの出力 -----
  
  # グラフを並べて描画
  comb_graph <- cowplot::plot_grid(
    model_graph, posterior_graph, predict_graph, 
    nrow = 3, ncol = 1, 
    align = "hv" # (目盛の共通化用)
  )
  comb_graph
  
  # 画像ファイルを書出
  file_path <- paste0(dir_path, "/", stringr::str_pad(n, width = nchar(N), pad = "0"), ".png")
  ggplot2::ggsave(
    filename = file_path, plot = comb_graph, 
    width = 9, height = 12, units = "in", dpi = 100
  )
  
  # 途中経過を表示
  message("\r", n, " / ", N, appendLF = FALSE)
}

# 動画を作成
paste0(dir_path, "/", stringr::str_pad(0:N, width = nchar(N), pad = "0"), ".png") |> # ファイルパスを作成
  magick::image_read() |> # pngファイルを読込
  magick::image_animate(fps = 1, dispose = "previous") |> # gifファイルを作成
  magick::image_write_video(path = "figure/poisson/parameter_updates/observation.mp4", framerate = 10) -> tmp_path # mp4ファイルを書出


