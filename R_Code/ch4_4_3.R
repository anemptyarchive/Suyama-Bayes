
# ch4.4.2 ガウス混合モデルにおけるギブスサンプリング -------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)


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
N <- 500
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

# 事前分布のパラメータを指定
beta <- 1
m_d <- rep(0, D)
nu <- D
w_dd <- diag(D) * 0.05
alpha_k <- rep(1, K)

solve(matrix(c(10, 0, 0, 10), D, D)) / D # Sigmaを指定してw_ddの初期値を計算


# 試行回数を指定
MaxIter <- 300


# 近似事後分布の初期値をランダムに設定
beta_hat_k <- seq(0.1, 10, by = 0.1) %>% 
  sample(size = K, replace = TRUE)
m_hat_kd <- seq(min(x_nd), max(x_nd), by = 0.1) %>% 
  sample(size = K * D, replace = TRUE) %>% 
  matrix(nrow = K, ncol = D)
#m_hat_kd <- matrix(0, nrow = K, ncol = D)
nu_hat_k <- rep(nu, K)
w_hat_ddk <- array(rep(w_dd, times = K), dim = c(D, D, K))
alpha_hat_k <- seq(0.1, 10, by = 0.1) %>% 
  sample(size = K, replace = TRUE)

# 途中計算に用いる項の受け皿を作成
ln_eta_nk <- matrix(0, nrow = N, ncol = K)
tmp_eta_nk <- matrix(0, nrow = N, ncol = K)
E_lmd_ddk <- rep(nu_hat_k, each = D * D) * w_hat_ddk
E_ln_det_lmd_k <- rep(0, K)
E_lmd_mu_kd <- matrix(0, nrow = K, ncol = D)
E_mu_lmd_mu_k <- rep(0, K)
E_ln_pi_k <- rep(0, K)

# 推移の確認用の受け皿を作成
trace_E_mu_ikd <- array(0, dim = c(MaxIter+1, K, D))
trace_E_lambda_iddk <- array(0, dim = c(MaxIter+1, D, D, K))
trace_E_s_ink <- array(0, dim = c(MaxIter, N, K))
trace_E_mu_ikd[1, , ] <- m_hat_kd
trace_E_lambda_iddk[1, , , ] <- rep(nu_hat_k, each = D * D) * w_hat_ddk

# 変分推論
for(i in 1:MaxIter) {
  
  # Sの近似事後分布のパラメータを計算:式(4.109)
  for(k in 1:K) {
    E_lmd_ddk[, , k] <- nu_hat_k[k] * w_hat_ddk[, , k]
    E_ln_det_lmd_k[k] <- sum(digamma(0.5 * (nu_hat_k[k] + 1 - 1:D))) + D * log(2) + log(det(w_hat_ddk[, , k]))
    E_lmd_mu_kd[k, ] <- E_lmd_ddk[, , k] %*% matrix(m_hat_kd[k, ])
    E_mu_lmd_mu_k[k] <- t(m_hat_kd[k, ]) %*% matrix(E_lmd_mu_kd[k, ]) + D / beta_hat_k[k]
    E_ln_pi_k[k] <- digamma(alpha_hat_k[k]) - digamma(sum(alpha_hat_k))
    term_eta1_n <- diag(
      -0.5 * x_nd %*% E_lmd_ddk[, , k] %*% t(x_nd)
    )
    term_eta2_n <- x_nd %*% matrix(E_lmd_mu_kd[k, ]) %>% 
      as.vector()
    ln_eta_nk[, k] <- term_eta1_n + term_eta2_n - 0.5 * E_mu_lmd_mu_k[k] + 0.5 * E_ln_det_lmd_k[k] + E_ln_pi_k[k]
  }
  tmp_eta_nk <- exp(ln_eta_nk)
  eta_nk <- (tmp_eta_nk + 1e-7) / rowSums(tmp_eta_nk + 1e-7) # 正規化
  
  # Sの近似事後分布の期待値を計算:式(4.59)
  E_s_nk <- eta_nk
  
  for(k in 1:K) {
    # muの近似事後分布のパラメータを計算:式(4.114)
    beta_hat_k[k] <- sum(E_s_nk[, k]) + beta
    m_hat_kd[k, ] <- (colSums(E_s_nk[, k] * x_nd) + beta * m_d) / beta_hat_k[k]
    
    # lambdaの近似事後分布のパラメータを計算:式(4.118)
    nu_hat_k[k] <- sum(E_s_nk[, k]) + nu
    tmp_w1_dd <- t(E_s_nk[, k] * x_nd) %*% x_nd
    tmp_w2_dd <- beta * matrix(m_d) %*% t(m_d)
    tmp_w3_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(
      tmp_w1_dd + tmp_w2_dd - tmp_w3_dd + solve(w_dd)
    )
  }
  
  # piの近似事後分布のパラメータを計算:式(4.58)
  alpha_hat_k <- colSums(E_s_nk) + alpha_k
  
  # 観測モデルのパラメータの期待値を記録
  trace_E_mu_ikd[i+1, , ] <- m_hat_kd
  trace_E_lambda_iddk[i+1, , , ] <- rep(nu_hat_k, each = D * D) * w_hat_ddk
  trace_E_s_ink[i, , ] <- E_s_nk
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}
print(round(E_s_nk))

# 作図用のデータフレームを作成
model_df <- tibble()
sample_df <- tibble()
max_p_idx <- max.col(E_s_nk) # 確率の最大値のインデックスを取得
for(k in 1:K) {
  # 近似事後分布を計算
  tmp_model_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      as.matrix(point_df), 
      mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
    ), 
    cluster = as.factor(k)
  )
  model_df <- rbind(model_df, tmp_model_df)
  
  # 観測データのクラスタを抽出
  k_idx <- which(max_p_idx == k)
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
  labs(title = "Variational Inference", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:', MaxIter), 
       x = expression(x[1]), y = expression(x[2]))


# 作図用のデータフレームを作成
trace_E_mu_df <- tibble()
trace_E_lambda_df <- tibble()
for(k in 1:K) {
  for(d1 in 1:D) {
    # muの値を取得
    tmp_mu_df <- tibble(
      iteration = seq(0, MaxIter), 
      value = trace_E_mu_ikd[, k, d1], 
      label = as.factor(
        paste0("k=", k, ", d=", d1)
      )
    )
    trace_E_mu_df <- rbind(trace_E_mu_df, tmp_mu_df)
    
    for(d2 in 1:D) {
      # lambdaの値を取得
      tmp_lambda_df <- tibble(
        iteration = seq(0, MaxIter), 
        value = trace_E_lambda_iddk[, d1, d2, k], 
        label = as.factor(
          paste0("k=", k, ", d=", d1, ", d'=", d2)
        )
      )
      trace_E_lambda_df <- rbind(trace_E_lambda_df, tmp_lambda_df)
    }
  }
}

# muの推移を確認
ggplot(trace_E_mu_df, aes(x = iteration, y = value, color = label)) + 
  geom_line() + 
  labs(title = "Variational Inference", 
       subtitle = expression(paste(E, "[", bold(mu), "]", sep = "")))

# lambdaの推移を確認
ggplot(trace_E_lambda_df, aes(x = iteration, y = value, color = label)) + 
  geom_line() + 
  labs(title = "Variational Inference", 
       subtitle = expression(paste(E, "[", bold(Lambda), "]", sep = "")))


# gif画像で推移を確認 -------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# 描画する回数を指定(変更)
#MaxIter <- 150

# 作図用のデータフレームを作成
model_df <- tibble()
sample_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # 確率の最大値のインデックスを取得
  if(i > 1) {
    max_p_idx <- max.col(trace_E_s_ink[i - 1, , ])
  }
  for(k in 1:K) {
    # 近似事後分布を計算
    tmp_model_df <- cbind(
      point_df, 
      density = mvnfast::dmvn(
        as.matrix(point_df), 
        mu = trace_E_mu_ikd[i, k, ], sigma = solve(trace_E_lambda_iddk[i, , , k])
      ), 
      cluster = as.factor(k), 
      iteration = as.factor(i - 1)
    )
    model_df <- rbind(model_df, tmp_model_df)
    
    # クラスタを抽出
    if(i > 1) { # 初期値以外のとき
      k_idx <- which(max_p_idx == k)
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
  labs(title = "Variational Inference", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:{current_frame}'), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
animate(trace_graphe, nframes = MaxIter, fps = 10)



# 散布図をグラデーションにしたい：未完 ------------------------------------------------------


# 散布図をグラデーションで表現
sample_df <- tibble(
  x1 = x_nd[, 1], 
  x2 = x_nd[, 2], 
  prob = E_s_nk[, 1]
)
ggplot() + 
  geom_point(data = sample_df, aes(x1, x2, color = prob)) + # 観測データ
  labs(title = "Variational Inference", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:', MaxIter), 
       x = expression(x[1]), y = expression(x[2]))


# gif ver
sample_df <- tibble()
for(i in 1:(MaxIter)) {
  
  tmp_sample_df <- tibble(
    x1 = x_nd[, 1], 
    x2 = x_nd[, 2], 
    prob = trace_E_s_ink[i, , 1], 
    iteration = as.factor(i)
  )
  sample_df <- rbind(sample_df, tmp_sample_df)

  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}

trace_graphe <- ggplot() + 
  geom_point(data = sample_df, aes(x1, x2, color = prob)) + # 観測データ
  transition_manual(iteration) + # フレーム
  labs(title = "Variational Inference", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:{current_frame}'), 
       x = expression(x[1]), y = expression(x[2]))

animate(trace_graphe, nframes = MaxIter, fps = 10)

