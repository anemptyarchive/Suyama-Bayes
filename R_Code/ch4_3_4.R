# 4.3.3 ポアソン混合モデルにおける崩壊型ギブスサンプリング -------------------------------------------------

# 4.3.3項で利用するパッケージ
library(tidyverse)
library(gganimate)


### 観測モデル(ポアソン混合分布)の設定 -----

# 真のパラメータを指定
lambda_truth_k <- c(5, 25, 50)

# 真の混合比率を指定
pi_truth_k <- c(0.35, 0.25, 0.4)

# クラスタ数を取得
K <- length(lambda_truth_k)


# 作図用のxの点を作成
x_vec <- seq(0, 2 * max(lambda_truth_k))

# 観測モデルを計算
model_prob <- 0
for(k in 1:K) {
  # クラスタkの分布の確率を計算
  tmp_prob <- dpois(x = x_vec, lambda_truth_k[k])
  
  # K個の分布の加重平均を計算
  model_prob <- model_prob + tmp_prob * pi_truth_k[k]
}

# 観測モデルをデータフレームに格納
model_df <- tibble(
  x = x_vec, 
  prob = model_prob
)

# 観測モデルを作図
ggplot(model_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", 
           fill = "purple", color = "purple") + # 真の分布
  labs(title = "Poisson Mixture Model", 
       subtitle = paste0(
         "lambda=(", paste0(lambda_truth_k, collapse = ", "), ")"
       ))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 250

# クラスタを生成
s_truth_nk <- rmultinom(n =  N, size = 1, prob = pi_truth_k) %>% 
  t()

# クラスタ番号を抽出
s_truth_n <- which(t(s_truth_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]

# (観測)データを生成
x_n <- rpois(n = N, lambda = apply(lambda_truth_k^t(s_truth_nk), 2, prod))
x_n <- rpois(n = N, lambda = lambda_truth_k[s_truth_n])

# 観測データを確認
summary(x_n)


# 観測データをデータフレームに格納
x_df <- tibble(
  x_n = x_n, 
  cluster = as.factor(s_truth_n)
)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_histogram(data = x_df, aes(x = x_n, y = ..density..), binwidth = 1) + # 観測データ
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Poisson Mixture Model", 
       subtitle = paste0("N=", N, ", lambda=(", paste0(lambda_truth_k, collapse = ", "), ")"), 
       x = "x")

# クラスタのヒストグラムを作成
ggplot() + 
  geom_histogram(data = x_df, aes(x = x_n, fill = cluster), binwidth = 1, 
                 position = "identity", alpha = 0.5) + # クラスタ
  labs(title = "Poisson Mixture Model", 
       subtitle = paste0("N=", N, 
                         ", lambda=(", paste0(lambda_truth_k, collapse = ", "), ")", 
                         ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"), 
       x = "x")


### 事前分布(ガンマ分布とディリクレ分布)の設定 -----

# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1

# piの事前分布のパラメータを指定
alpha_k <- rep(2, K)


### 初期値の設定 -----

# 潜在変数の初期値をランダムに生成
s_nk <- rmultinom(n = N, size = 1, prob = alpha_k / sum(alpha_k)) %>% 
  t()

# lambdaの事後分布のパラメータを計算:式(4.24)
a_hat_k <- colSums(s_nk * x_n) + a
b_hat_k <- colSums(s_nk) + b

# piの事後分布のパラメータを計算:式(4.45)
alpha_hat_k <- colSums(s_nk) + alpha_k


# 初期値による混合分布を計算
init_prob <- 0
for(k in 1:K) {
  # クラスタkの分布の確率を計算
  tmp_prob <- dpois(x = x_vec, a_hat_k[k] / b_hat_k[k])
  
  #　K個の分布の加重平均を計算
  init_prob <- init_prob + tmp_prob * alpha_hat_k[k] / sum(alpha_hat_k)
}

# 初期値による分布をデータフレームに格納
init_df <- tibble(
  x = x_vec, 
  prob = init_prob
)

# 初期値による分布を作図
ggplot(init_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity") + # 初期値による分布
  labs(title = "Poisson Mixture Model", 
       subtitle = paste0("iter:", 0, 
                         ", E[lambda]=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")"))


### 推論処理 -----

# 試行回数を指定
MaxIter <- 100

# 推移の確認用の受け皿を初期化
trace_s_in <- matrix(0, nrow = MaxIter + 1, ncol = N)
trace_a_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_b_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_alpha_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)

# 初期値を記録
trace_s_in[1, ] <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]
trace_a_ik[1, ] <- a_hat_k
trace_b_ik[1, ] <- b_hat_k
trace_alpha_ik[1, ] <- alpha_hat_k

# 崩壊型ギブスサンプリング
for(i in 1:MaxIter) {
  for(n in 1:N) {
    
    # n番目のデータの現在のクラスタ番号を取得
    k <- which(s_nk[n, ] == 1)
    
    # n番目のデータに関する統計量を除算
    a_hat_k[k] <- a_hat_k[k] - x_n[n]
    b_hat_k[k] <- b_hat_k[k] - 1
    alpha_hat_k[k] <- alpha_hat_k[k] - 1
    
    # 負の二項分布(4.81)のパラメータを計算
    r_hat_k <- a_hat_k
    p_hat_k <- 1 / (b_hat_k + 1)
    
    # 負の二項分布の確率を計算:式(4.81)
    prob_nb_k <- dnbinom(x = x_n[n], size = r_hat_k, prob = 1 - p_hat_k)
    
    # カテゴリ分布(4.74)のパラメータを計算:式(4.75)
    eta_k <- alpha_hat_k / sum(alpha_hat_k)
    
    # n番目のクラスタのサンプリング確率を計算:式(4.66)
    prob_s_k <- (prob_nb_k + 1e-7) * eta_k
    prob_s_k <- prob_s_k / sum(prob_s_k) # 正規化
    
    # n番目のクラスタをサンプル:式(4.74)
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = prob_s_k) %>% 
      as.vector()
    
    # n番目のデータの新しいクラスタを取得
    k <- which(s_nk[n, ] == 1)
    
    # n番目のデータに関する統計量を加算:式(4.82-4.83)
    a_hat_k[k] <- a_hat_k[k] + x_n[n]
    b_hat_k[k] <- b_hat_k[k] + 1
    alpha_hat_k[k] <- alpha_hat_k[k] + 1
  }
  
  # i回目の結果を記録
  trace_s_in[i + 1, ] <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"] %>% 
    as.vector()
  trace_a_ik[i + 1, ] <- a_hat_k
  trace_b_ik[i + 1, ] <- b_hat_k
  trace_alpha_ik[i + 1, ] <- alpha_hat_k
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 結果の確認 -----

# 最後の分布を計算
res_prob <- 0
for(k in 1:K) {
  # クラスタkの分布の確率を計算
  tmp_prob <- dpois(x = x_vec, lambda = a_hat_k[k] / b_hat_k[k])
  
  # K個の分布の加重平均を計算
  res_prob <- res_prob + tmp_prob * alpha_hat_k[k] / sum(alpha_hat_k)
}

# 最後の分布をデータフレームに格納
res_df <- tibble(
  x = x_vec, 
  prob = res_prob
)

# 最後の分布を作図
ggplot() + 
  geom_bar(data = res_df, aes(x = x, y = prob), stat = "identity", 
           fill = "purple") + # 最後の分布
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, 
                         ", E[lambda]=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")"))


# 最後のクラスタをデータフレームに格納
s_df <- tibble(
  x_n = x_n, 
  cluster = which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"] %>% 
    as.factor()
)

# 最後のクラスタのヒストグラムを作成
ggplot() + 
  geom_histogram(data = s_df, aes(x = x_n, fill = cluster), binwidth = 1, position = "identity", 
                 alpha = 0.5) + # 最後のクラスタ
  geom_histogram(data = x_df, aes(x = x_n, color = cluster), binwidth = 1, position = "identity",
                 alpha = 0, linetype = "dashed") + # 真のクラスタ
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, 
                         ", E[lambda]=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")", 
                         ", E[pi]=(", paste0(round(alpha_hat_k / sum(alpha_hat_k), 2), collapse = ", "), ")"), 
       x = "x")


### 超パラメータの推移の確認 -----

# aの推移をデータフレームに格納
trace_a_df <- dplyr::as_tibble(trace_a_ik) %>% # データフレームに変換
  cbind(iteration = 0:MaxIter) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) # 縦持ちに変換

# aの推移を作図
ggplot(trace_a_df, aes(x = iteration, y = value, color = cluster)) + 
  geom_line() + 
  labs(title = "Variational Inferance", 
       subtitle = expression(hat(bold(a))))

# bの推移をデータフレームに格納
trace_a_df <- dplyr::as_tibble(trace_b_ik) %>% # データフレームに変換
  cbind(iteration = 0:MaxIter) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) # 縦持ちに変換

# bの推移を作図
ggplot(trace_a_df, aes(x = iteration, y = value, color = cluster)) + 
  geom_line() + 
  labs(title = "Variational Inferance", 
       subtitle = expression(hat(bold(b))))

# alphaの推移を作図
trace_alpha_df <- dplyr::as_tibble(trace_alpha_ik) %>% # データフレームに変換
  cbind(iteration = 0:MaxIter) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) # 縦持ちに変換

# alphaの推移を作図
ggplot(trace_alpha_df, aes(x = iteration, y = value, color = cluster)) + 
  geom_line() + 
  labs(title = "Variational Inferance", 
       subtitle = expression(hat(alpha)))


### 分布の推移の確認 -----

# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # i回目の分布を計算
  res_prob <- 0
  for(k in 1:K) {
    # クラスタkの分布の確率を計算
    tmp_prob <- dpois(x = x_vec, lambda = trace_a_ik[i, k] / trace_b_ik[i, k])
    
    # K個の分布の加重平均を計算
    res_prob <- res_prob + tmp_prob * trace_alpha_ik[i, k] / sum(trace_alpha_ik[i, ])
  }
  
  # i回目の分布をデータフレームに格納
  res_df <- tibble(
    x = x_vec, 
    prob = res_prob, 
    label = paste0(
      "iter:", i - 1, ", N=", N, 
      ", E[lambda]=(", paste0(round(trace_a_ik[i, ] / trace_b_ik[i, ], 2), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  
  # i回目のクラスタをデータフレームに格納
  s_df <- tibble(
    x_n = x_n, 
    cluster = as.factor(trace_s_in[i, ]), 
    label = paste0(
      "iter=", i - 1, ", N=", N, 
      ", E[lambda]=(", paste0(round(trace_a_ik[i, ] / trace_b_ik[i, ], 2), collapse = ", "), ")", 
      ", E[pi]=(", paste0(round(trace_alpha_ik[i, ] / sum(trace_alpha_ik[i, ]), 2), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 結果を結合
  trace_cluster_df <- rbind(trace_cluster_df, s_df)
}

# アニメーション用に複製
rep_model_df <- tibble(
  x = rep(model_df[["x"]], times = MaxIter + 1), 
  prob = rep(model_df[["prob"]], times = MaxIter + 1), 
  label = trace_model_df[["label"]]
)

# 分布の推移を作図
trace_graph <- ggplot() + 
  geom_bar(data = trace_model_df, aes(x = x, y = prob), stat = "identity", 
           fill = "purple") + # 推定した分布
  geom_bar(data = rep_model_df, aes(x = x, y = prob), stat = "identity", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = "{current_frame}")

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


# アニメーション用に複製
rep_x_df <- tibble(
  x_n = rep(x_n, times = MaxIter + 1), 
  cluster = rep(as.factor(s_truth_n), times = MaxIter + 1), 
  label = trace_cluster_df[["label"]]
)

# クラスタの推移を作図
trace_graph <- ggplot() + 
  geom_histogram(data = trace_cluster_df, aes(x = x_n, fill = cluster), 
                 binwidth = 1, position = "identity",
                 alpha = 0.5) + # 最後のクラスタ
  geom_histogram(data = rep_x_df, aes(x = x_n, color = cluster), 
                 binwidth = 1, position = "identity",
                 alpha = 0, linetype = "dashed") + # 真のクラスタ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = "{current_frame}", 
       x = "x")

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


