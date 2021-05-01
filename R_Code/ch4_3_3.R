
# 4.3.3 ポアソン混合モデルにおける推論：変分推論 -------------------------------------------------

# 4.4.3項で利用するパッケージ
library(tidyverse)
library(gganimate)


### 観測モデル(ポアソン混合分布)の設定 -----

# 真のパラメータを指定
lambda_truth_k <- c(5, 20, 35)

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
       subtitle = paste0("lambda=(", paste0(lambda_truth_k, collapse = ", "), ")"))


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


# 事前分布の平均による分布をデータフレームに格納
E_prior_df <- tibble(
  x = x_vec, 
  prob = dpois(x = x_vec, a / b)
)

# 事前分布の平均による分布を作図
ggplot(E_prior_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", 
           fill = "purple", color = "purple") + # 初期値による分布
  labs(title = "Poisson Distribution", 
       subtitle = paste0("iter:", 0, ", lambda=", round(a / b, 2)))


### 初期値の設定 -----

# クラスタの事後分布の期待値の初期値を生成
E_s_nk <- seq(0, 1, by = 0.01) %>% 
  sample(size = N * K, replace = TRUE) %>% 
  matrix(nrow = N, ncol = K)
E_s_nk <- E_s_nk / rowSums(E_s_nk) # 正規化

# 初期化によるlambdaの事後分布のパラメータを計算:式(4.55)
a_hat_k <- colSums(E_s_nk * x_n) + a
b_hat_k <- colSums(E_s_nk) + b

# 初期値によるpiのパラメータを計算:式(4.58)
alpha_hat_k <- colSums(E_s_nk) + alpha_k


# 初期値によるlambdaの近似事後分布をデータフレームに格納
init_lambda_df <- tibble()
for(k in 1:K) {
  # クラスタkの事後分布を計算
  tmp_init_df <- tibble(
    lambda = seq(0, 2 * max(lambda_truth_k), length.out = 1000), # 作図用のlambdaの点
    density = dgamma(x = lambda, shape = a_hat_k[k], rate = b_hat_k[k]), 
    cluster = as.factor(k)
  )
  
  # 結果を結合
  init_lambda_df <- rbind(init_lambda_df, tmp_init_df)
}

# 初期値によるlambdaの近似事後分布を作図
ggplot(init_lambda_df, aes(x = lambda, y = density, color = cluster)) + 
  geom_line() + # 初期値による事後分布
  geom_vline(xintercept = lambda_truth_k, color = "red", linetype = "dashed") + # 真の値
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", 0, 
                         ", a=(", paste0(round(a_hat_k, 1), collapse = ", "), ")", 
                         ", b=(", paste0(round(b_hat_k, 1), collapse = ", "), ")"), 
       x = expression(lambda))


# 初期値による混合分布を計算
init_prob <- 0
for(k in 1:K) {
  # クラスタkの分布の確率を計算
  tmp_prob <- dpois(x = x_vec, a_hat_k[k] / b_hat_k[k])
  
  # K個の分布の加重平均を計算
  init_prob <- init_prob + tmp_prob * alpha_hat_k[k] / sum(alpha_hat_k)
}

# 初期値による分布をデータフレームに格納
init_df <- tibble(
  x = x_vec, 
  prob = init_prob
)

# 初期値による分布を作図
ggplot() + 
  geom_bar(data = init_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           fill = "purple", color = "purple") + # 初期値による分布
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", 0, 
                         ", lambda=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")"))


# クラスタの初期値をデータフレームに格納
cluster_df <- tibble(
  x = x_n, 
  cluster = max.col(E_s_nk) %>% 
    as.factor(), 
  prob = E_s_nk[cbind(1:N, max.col(E_s_nk))]
) %>% 
  dplyr::group_by(x, cluster, prob) %>% 
  dplyr::summarise(count = dplyr::n())

# クラスタの初期値のヒストグラムを作成
ggplot() + 
  geom_bar(data = cluster_df, aes(x = x, y = count, fill = cluster, color = cluster), stat = "identity", position = "identity", 
           alpha = cluster_df[["prob"]]) + # クラスタの初期値
  geom_histogram(data = x_df, aes(x = x_n, color = cluster), binwidth = 1, position = "identity",
                 alpha = 0, linetype = "dashed") + # 真のクラスタ
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", 0, 
                         ", lambda=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")", 
                         ", pi=(", paste0(round(alpha_hat_k / sum(alpha_hat_k), 2), collapse = ", "), ")"), 
       x = "x")


### 推論処理 -----

# 試行回数を指定
MaxIter <- 50

# 変数を作成
eta_nk <- matrix(0, nrow = N, ncol = K)

# 推移の確認用の受け皿を作成
trace_E_s_ink <- array(0, dim = c(MaxIter + 1, N, K))
trace_a_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_b_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_alpha_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)

# 初期値を記録
trace_E_s_ink[1, , ] <- E_s_nk
trace_a_ik[1, ] <- rep(a, K)
trace_b_ik[1, ] <- rep(a, K)
trace_alpha_ik[1, ] <- alpha_k

# 変分推論
for(i in 1:MaxIter) {
  for(n in 1:N) {
    
    # 潜在変数の事後分布のパラメータの各項を計算:式(4.60-4.62)
    E_lmd_k <- a_hat_k / b_hat_k
    E_ln_lmd_k <- digamma(a_hat_k) - log(b_hat_k)
    E_ln_pi_k <- digamma(alpha_hat_k) - digamma(sum(alpha_hat_k))
    
    # 潜在変数の事後分布のパラメータを計算:式(4.51)
    tmp_eta_k <- exp(x_n[n] * E_ln_lmd_k - E_lmd_k + E_ln_pi_k)
    eta_nk[n, ] <- tmp_eta_k / sum(tmp_eta_k) # 正規化
    
    # 潜在変数の事後分布の期待値を計算:式(4.59)
    E_s_nk[n, ] <- eta_nk[n, ]
  }
  
  # lambdaの事後分布のパラメータを計算:式(4.55)
  a_hat_k <- colSums(E_s_nk * x_n) + a
  b_hat_k <- colSums(E_s_nk) + b
  
  # piのパラメータを計算:式(4.58)
  alpha_hat_k <- colSums(E_s_nk) + alpha_k
  
  # i回目の結果を記録
  trace_E_s_ink[i + 1, , ] <- E_s_nk
  trace_a_ik[i + 1, ] <- a_hat_k
  trace_b_ik[i + 1, ] <- b_hat_k
  trace_alpha_ik[i + 1, ] <- alpha_hat_k
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### パラメータの近似事後分布の確認 -----

# lambdaの近似事後分布をデータフレームに格納
posterior_lambda_df <- tibble()
for(k in 1:K) {
  # クラスタkの事後分布を計算
  tmp_posterior_df <- tibble(
    lambda = seq(0, 2 * max(lambda_truth_k), length.out = 1000), # 作図用のlambdaの点
    density = dgamma(x = lambda, shape = a_hat_k[k], rate = b_hat_k[k]), 
    cluster = as.factor(k)
  )
  
  # 結果を結合
  posterior_lambda_df <- rbind(posterior_lambda_df, tmp_posterior_df)
}

# lambdaの近似事後分布を作図
ggplot(posterior_lambda_df, aes(x = lambda, y = density, color = cluster)) + 
  geom_line() + # lambdaの事後分布
  geom_vline(xintercept = lambda_truth_k, color = "red", linetype = "dashed") + # 真の値
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, 
                         ", a=(", paste0(round(a_hat_k, 1), collapse = ", "), ")", 
                         ", b=(", paste0(round(b_hat_k, 1), collapse = ", "), ")"), 
       x = expression(lambda))


### 最後の推定値の確認 -----

# 最後の推定値による混合分布を計算
res_prob <- 0
for(k in 1:K) {
  # クラスタkの分布の確率を計算
  tmp_prob <- dpois(x = x_vec, a_hat_k[k] / b_hat_k[k])
  
  # K個の分布の加重平均を計算
  res_prob <- res_prob + tmp_prob * alpha_hat_k[k] / sum(alpha_hat_k)
}

# 最後のサンプルによる分布をデータフレームに格納
res_df <- tibble(
  x = x_vec, 
  prob = res_prob
)

# 最後の推定値による分布を作図
ggplot() + 
  geom_bar(data = res_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           fill = "purple", color = "purple") + # 最後の推定値による分布
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, 
                         ", lambda=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")"))


# 最後のクラスタをデータフレームに格納
cluster_df <- tibble(
  x = x_n, 
  cluster = max.col(E_s_nk) %>% 
    as.factor(), 
  prob = E_s_nk[cbind(1:N, max.col(E_s_nk))]
) %>% 
  dplyr::group_by(x, cluster, prob) %>% 
  dplyr::summarise(count = dplyr::n())

# 最後のクラスタのヒストグラムを作成
ggplot() + 
  geom_bar(data = cluster_df, aes(x = x, y = count, fill = cluster, color = cluster), stat = "identity", position = "identity", 
           alpha = cluster_df[["prob"]]) + # 最後のクラスタ
  geom_histogram(data = x_df, aes(x = x_n, color = cluster), binwidth = 1, position = "identity",
                 alpha = 0, linetype = "dashed") + # 真のクラスタ
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, 
                         ", lambda=(", paste0(round(a_hat_k / b_hat_k, 2), collapse = ", "), ")", 
                         ", pi=(", paste0(round(alpha_hat_k / sum(alpha_hat_k), 2), collapse = ", "), ")"), 
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
  labs(title = "Gibbs Sampling", 
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
  labs(title = "Gibbs Sampling", 
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
  labs(title = "Gibbs Sampling", 
       subtitle = expression(hat(alpha)))


# ・アニメーションによる確認 ----------------------------------------------------------

# 追加パッケージ
library(gganimate)


### 近似事後分布の推移の確認 -----

# 作図用のlambdaの点を作成
lambda_vec <- seq(0, 2 * max(lambda_truth_k), length.out = 1000)

# 作図用のデータフレームを作成
trace_posterior_lambda_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  for(k in 1:K) {
    # クラスタkの事後分布を計算
    tmp_posterior_df <- tibble(
      lambda = lambda_vec, 
      density = dgamma(x = lambda, shape = trace_a_ik[i, k], rate = trace_b_ik[i, k]), 
      cluster = as.factor(k), 
      label = paste0(
        "iter:", i - 1, ", N=", N, 
        ", a=(", paste0(round(trace_a_ik[i, ], 1), collapse = ", "), ")", 
        ", b=(", paste0(round(trace_b_ik[i, ], 1), collapse = ", "), ")"
      ) %>% 
        as.factor()
    )
    
    # 結果を結合
    trace_posterior_lambda_df <- rbind(trace_posterior_lambda_df, tmp_posterior_df)
  }
  
  # 動作確認
  print(paste0((i - 1), ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}

# lambdaの事後分布を作図
trace_graph <- ggplot() + 
  geom_line(data = trace_posterior_lambda_df, aes(x = lambda, y = density, color = cluster)) + # lambdaの事後分布
  geom_vline(xintercept = lambda_truth_k, color = "red", linetype = "dashed") + # 真の値
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Poisson mixture model:Variational Inference", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


### サンプルの推移の確認 -----

# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # i回目の混合分布を計算
  res_prob <- 0
  for(k in 1:K) {
    # クラスタkの分布の確率を計算
    tmp_prob <- dpois(x = x_vec, trace_a_ik[i, k] / trace_b_ik[i, k])
    
    # K個の分布の加重平均を計算
    res_prob <- res_prob + tmp_prob * trace_alpha_ik[i, k] / sum(trace_alpha_ik[i, ])
  }
  
  # i回目のサンプルによる分布をデータフレームに格納
  res_df <- tibble(
    x = x_vec, 
    prob = res_prob, 
    label = paste0(
      "iter:", i - 1, ", N=", N, 
      ", E[lambda]=(", paste0(round(trace_a_ik[i, ] / trace_b_ik[i, ], 2), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # i回目のクラスタをデータフレームに格納
  cluster_df <- tibble(
    x = x_n, 
    cluster = max.col(trace_E_s_ink[i, , ]) %>% 
      as.factor(), 
    prob = E_s_nk[cbind(1:N, max.col(trace_E_s_ink[i, , ]))], 
    label = paste0(
      "iter=", i - 1, ", N=", N, 
      ", E[lambda]=(", paste0(round(trace_a_ik[i, ] / trace_b_ik[i, ], 2), collapse = ", "), ")", 
      ", E[pi]=(", paste0(round(trace_alpha_ik[i, ] / sum(trace_alpha_ik[i, ]), 2), collapse = ", "), ")"
    ) %>% 
      as.factor()
  ) %>% 
    dplyr::group_by(x, cluster, prob, label) %>% 
    dplyr::summarise(count = dplyr::n())
  
  # i回目の結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  trace_cluster_df <- rbind(trace_cluster_df, cluster_df)
  
  # 動作確認
  print(paste0((i - 1), ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = "{current_frame}")

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


# アニメーション用に複製
rep_x_df <- tibble(
  x_n = rep(x_n, times = MaxIter + 1), 
  cluster = rep(as.factor(s_truth_n), times = MaxIter + 1), 
  label = unique(trace_cluster_df[["label"]]) %>% 
    rep(each = N)
)

# クラスタの推移を作図
trace_graph <- ggplot() + 
  geom_bar(data = trace_cluster_df, aes(x = x, y = count, color = cluster, fill = cluster), stat = "identity", position = "identity", 
           alpha = trace_cluster_df[["prob"]]) + # 推定したクラスタ
  geom_histogram(data = rep_x_df, aes(x = x_n, color = cluster), 
                 binwidth = 1, position = "identity",
                 alpha = 0, linetype = "dashed") + # 真のクラスタ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = "{current_frame}", 
       x = "x")

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


