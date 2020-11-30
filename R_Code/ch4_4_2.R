
# ch4.4.2 ガウス混合モデルにおけるギブスサンプリング -------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)
library(MCMCpack)


# モデルの設定 ------------------------------------------------------------------

# 真の観測モデルのパラメータを指定
D <- 2 # (固定)
K <- 3
mu_true_kd <- matrix(
  c(0, 4, 
    -5, -5, 
    5, -2.5), nrow = K, ncol = D, byrow = TRUE
)
sigma2_true_ddk <- array(
  c(8, 0, 0, 8, 
    4, -2.5, -2.5, 4, 
    6.5, 4, 4, 6.5), dim = c(D, D, K)
)

# 真の混合比率を指定
pi_true_k <- c(0.5, 0.2, 0.3)


# 観測データの真のクラスタを生成
N <- 250
s_true_nk <- rmultinom(n = N, size = 1, prob = pi_true_k) %>% 
  t()
res_s <- which(t(s_true_nk) == 1, arr.ind = TRUE)
s_true_n <- res_s[, "row"]

# 観測データを生成
x_nd <- matrix(0, nrow = N, ncol = D)
for(n in 1:N) {
  k <- s_true_n[n] # クラスタを取得
  x_nd[n, ] = mvnfast::rmvn(n = 1, mu = mu_true_kd[k, ], sigma = sigma2_true_ddk[, , k])
}


# 作図用の点を生成
x_line <- seq(-10, 10, by = 0.1)
point_df <- tibble(
  x1 = rep(x_line, times = length(x_line)), 
  x2 = rep(x_line, each = length(x_line))
)

# 作図用のデータフレームを作成
model_true_df <- tibble()
sample_df <- tibble()
for(k in 1:K) {
  # 真の観測モデルを計算
  tmp_model_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      X = as.matrix(point_df), mu = mu_true_kd[k, ], sigma = sigma2_true_ddk[, , k]
    ), 
    cluster = as.factor(k)
  )
  model_true_df <- rbind(model_true_df, tmp_model_df)
  
  # 観測データのデータフレーム
  k_idx <- which(s_true_n == k)
  tmp_sample_df <- tibble(
    x1 = x_nd[k_idx, 1], 
    x2 = x_nd[k_idx, 2], 
    cluster = as.factor(k)
  )
  sample_df <- rbind(sample_df, tmp_sample_df)
}

# 真の観測モデルを作図
ggplot() + 
  geom_contour(data = model_true_df, aes(x1, x2, z = density, color = cluster)) + # 真の観測モデル
  geom_point(data = sample_df, aes(x1, x2, color = cluster)) + # 真の観測データ
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('K=', K, ', N=', N), 
       x = expression(x[1]), y = expression(x[2]))


# 推論処理 --------------------------------------------------------------------

# 観測モデルのパラメータの初期値を指定
mu_kd <- matrix(0, nrow = K, ncol = D)
solve(diag(D) * 100)
lambda_ddk <- array(
  c(0.01, 0, 0, 0.01, 
    0.01, 0, 0, 0.01, 
    0.01, 0, 0, 0.01), dim = c(D, D, K)
)

# 混合比率の初期値を指定
pi_k <- sample(seq(0, 1, by = 0.01), size = K)
pi_k <- pi_k / sum(pi_k) # 正規化


# 事前分布のパラメータを指定
beta <- 1
m_d <- rep(0, D)
nu <- D
w_dd <- diag(D) * 10
alpha_k <- rep(1, K)

# 試行回数を指定
MaxIter <- 100

# 推移の確認用の受け皿
trace_s_in <- matrix(0, nrow = MaxIter, ncol = N)
trace_mu_ikd <- array(0, dim = c(MaxIter+1, K, D))
trace_lambda_iddk <- array(0, dim = c(MaxIter+1, D, D, K))
trace_mu_ikd[1, , ] <- mu_kd
trace_lambda_iddk[1, , , ] <- lambda_ddk

# ギブスサンプリング
for(i in 1:MaxIter) {
  
  # 初期化
  eta_nk <- matrix(0, nrow = N, ncol = K)
  s_nk <- matrix(0, nrow = N, ncol = K)
  beta_hat_k <- rep(0, K)
  m_hat_kd <- matrix(0, nrow = K, ncol = D)
  nu_hat_k <- rep(0, K)
  w_hat_ddk <- array(0, dim = c(D, D, K))
  alpha_hat_k <- rep(0, K)
  
  # 潜在変数変数のパラメータを計算:式(4.94)
  for(k in 1:K) {
    tmp_term_dn <- t(x_nd) - mu_kd[k, ]
    tmp_eta_n <- diag(
      t(tmp_term_dn) %*% lambda_ddk[, , k] %*% tmp_term_dn
    )
    tmp_eta <- 0.5 * log(det(lambda_ddk[, , k]) + 1e-7) + log(pi_k[k] + 1e-7)
    eta_nk[, k] <- exp(-0.5 * tmp_eta_n + tmp_eta)
  }
  eta_nk <- eta_nk / rowSums(eta_nk) # 正規化
  
  # 潜在変数をサンプル
  for(n in 1:N) {
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
  }
  
  # 観測モデルのパラメータをサンプル
  for(k in 1:K) {
    
    # muの事後分布のパラメータを計算:式(4.99)
    beta_hat_k[k] <- sum(s_nk[, k]) + beta
    m_hat_kd[k, ] <- (colSums(s_nk[, k] * x_nd) + beta * m_d) / beta_hat_k[k]
    
    # lambdaの事後分布のパラメータを計算:式(4.103)
    nu_hat_k[k] <- sum(s_nk[, k]) + nu
    tmp_w1_dd <- t(s_nk[, k] * x_nd) %*% x_nd
    tmp_w2_dd <- beta * matrix(m_d) %*% t(m_d)
    tmp_w3_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(
      tmp_w1_dd + tmp_w2_dd - tmp_w3_dd + solve(w_dd)
    )
    
    # lambdaをサンプル:式(4.102)
    lambda_ddk[, , k] <- rWishart(n = 1, df = nu_hat_k[k], Sigma = w_hat_ddk[, , k])
    #lambda_ddk[, , k] <- MCMCpack::rwish(v = nu_hat_k[k], S = w_hat_ddk[, , k])
    
    # muをサンプル:式(4.98)
    mu_kd[k, ] <- mvnfast::rmvn(
      n = 1, mu = m_hat_kd[k, ], sigma = solve(beta_hat_k[k] * lambda_ddk[, , k])
    ) %>% 
      as.vector()
  }
  
  # 混合比率のパラメータを計算:式(4.45)
  alpha_hat_k <- colSums(s_nk) + alpha_k
  
  # piをサンプル:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_hat_k) %>% 
    as.vector()
  
  # 値を記録
  res_s <- which(t(s_nk) == 1, arr.ind = TRUE)
  trace_s_in[i, ] <- res_s[, "row"]
  trace_mu_ikd[i+1, , ] <- mu_kd
  trace_lambda_iddk[i+1, , , ] <- lambda_ddk
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


# 作図用のデータフレームを作成
model_df <- tibble()
sample_df <- tibble()
for(k in 1:K) {
  # 近似事後分布を計算
  tmp_model_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      as.matrix(point_df), mu = mu_kd[k, ], sigma = solve(lambda_ddk[, , k])
    ), 
    cluster = as.factor(k)
  )
  model_df <- rbind(model_df, tmp_model_df)
  
  # 観測データのクラスタを抽出
  k_idx <- which(s_nk[, k] == 1)
  tmp_sample_df <- tibble(
    x1 = x_nd[k_idx, 1], 
    x2 = x_nd[k_idx, 2], 
    cluster = as.factor(k)
  )
  sample_df <- rbind(sample_df, tmp_sample_df)
}

# 近似事後分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x1, x2, z = density, color = cluster)) + # 近似事後分布
  geom_contour(data = model_true_df, aes(x1, x2, z = density, color = cluster), 
               linetype = "dotted", alpha = 0.6) + # 真の観測モデル
  geom_point(data = sample_df, aes(x1, x2, color = cluster)) + # 観測データ
  labs(title = "Gibbs Sampling", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:', MaxIter), 
       x = expression(x[1]), y = expression(x[2]))


# 作図用のデータフレームを作成
trace_mu_df <- tibble()
trace_lambda_df <- tibble()
for(k in 1:K) {
  for(d1 in 1:D) {
    # muの値を取得
    tmp_mu_df <- tibble(
      iteration = seq(0, MaxIter), 
      value = trace_mu_ikd[, k, d1], 
      label = as.factor(
        paste0("k=", k, ", d=", d1)
      )
    )
    trace_mu_df <- rbind(trace_mu_df, tmp_mu_df)
    
    for(d2 in 1:D) {
      # lambdaの値を取得
      tmp_lambda_df <- tibble(
        iteration = seq(0, MaxIter), 
        value = trace_lambda_iddk[, d1, d2, k], 
        label = as.factor(
          paste0("k=", k, ", d=", d1, ", d'=", d2)
        )
      )
      trace_lambda_df <- rbind(trace_lambda_df, tmp_lambda_df)
    }
  }
}

# muの推移を確認
ggplot(trace_mu_df, aes(x = iteration, y = value, color = label)) + 
  geom_line() + 
  labs(title = expression(bold(mu)))

# lambdaの推移を確認
ggplot(trace_lambda_df, aes(x = iteration, y = value, color = label)) + 
  geom_line() + 
  labs(title = expression(bolditalic(Lambda)))


# gif画像で推移を確認 -------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
model_df <- tibble()
sample_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  for(k in 1:K) {
    # 観測モデルを計算
    tmp_model_df <- cbind(
      point_df, 
      density = mvnfast::dmvn(
        as.matrix(point_df), mu = trace_mu_ikd[i, k, ], sigma = solve(trace_lambda_iddk[i, , , k])
      ), 
      cluster = as.factor(k), 
      iteration = as.factor(i-1)
    )
    model_df <- rbind(model_df, tmp_model_df)
    
    # 観測データのデータフレーム
    if(i > 1) { # 初期値以外のとき
      k_idx <- which(trace_s_in[i - 1, ] == k)
      tmp_sample_df <- tibble(
        x1 = x_nd[k_idx, 1], 
        x2 = x_nd[k_idx, 2], 
        cluster = as.factor(k), 
        iteration = as.factor(i - 1)
      )
      sample_df <- rbind(sample_df, tmp_sample_df)
    }
  }
  
  if(i == 1) { # 初期値のとき
    tmp_sample_df <- tibble(
      x1 = x_nd[, 1], 
      x2 = x_nd[, 2], 
      cluster = NA, 
      iteration = as.factor(i - 1)
    )
    sample_df <- rbind(sample_df, tmp_sample_df)
  }
  
  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}

# 近似事後分布を作図
trace_graphe <- ggplot() + 
  geom_contour(data = model_df, aes(x1, x2, z = density, color = cluster)) + # 近似事後分布
  geom_contour(data = model_true_df, aes(x1, x2, z = density, color = cluster), 
               linetype = "dotted", alpha = 0.6) + # 真の観測モデル
  geom_point(data = sample_df, aes(x1, x2, color = cluster)) + # 観測データ
  transition_manual(iteration) + # フレーム
  labs(title = "Gibbs Sampling", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:{current_frame}'), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
animate(trace_graphe, nframes = MaxIter + 1, fps = 10)

