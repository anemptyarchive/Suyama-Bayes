
# 3.2.2 カテゴリ分布の学習と予測 -------------------------------------------------------------------

# 3.2.2 事後分布 -------------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメーターの初期値を指定
# 観測モデルのパラメータ
pi_k_truth <- c(0.3, 0.5, 0.2)

# 事前分布のパラメータ
alpha_k <- c(2, 2, 2)

# 試行回数
N <- 50


# 作図用のpiの値を満遍なく生成
pi <- tibble(
  pi_1 = rep(rep(seq(0, 1, by = 0.02), times = 51), times = 51), 
  pi_2 = rep(rep(seq(0, 1, by = 0.02), each = 51), times = 51), 
  pi_3 = rep(seq(0, 1, by = 0.02), each = 2601)
)

# 正規化
pi <- pi / apply(pi, 1, sum)

# 重複した組み合わせを除去(ハイスぺ機なら不要…)
pi <- pi %>% 
  mutate(pi_1 = round(pi_1, 3), pi_2 = round(pi_2, 3), pi_3 = round(pi_3, 3)) %>% 
  count(pi_1, pi_2, pi_3) %>% 
  select(-n) %>% 
  as.matrix()


# 作図用のpiの値をランダムに生成
#pi <- matrix(sample(seq(0, 1, 0.01), 90000, replace = TRUE), nrow = 3)

# 正規化
#pi <- pi / apply(pi, 1, sum)


# カテゴリ分布に従うデータを生成
s_nk <- rmultinom(n = N, size = 1, prob = pi_k_truth) %>% 
  t()

# 観測データを確認
apply(s_nk, 2, sum)


# 事後分布のパラメータを計算
alpha_k_hat <- apply(s_nk, 2, sum) + alpha_k


# 事後分布を計算
posterior_df <- tibble(
  x = pi[, 2] + (pi[, 3] / 2),  # 三角座標への変換
  y = sqrt(3) * (pi[, 3] / 2),  # 三角座標への変換
  C_D = lgamma(sum(alpha_k_hat)) - sum(lgamma(alpha_k_hat)),  # 正規化項(対数)
  density = exp(C_D + apply((alpha_k_hat - 1) * log(t(pi)), 2, sum)) # 確率密度
)


# piの真の値のプロット用データフレームを作成
pi_truth_df <- tibble(
  x = pi_k_truth[2] + (pi_k_truth[3] / 2),  # 三角座標への変換
  y = sqrt(3) * (pi_k_truth[3] / 2),  # 三角座標への変換
)

# 作図
ggplot() + 
  geom_point(data = posterior_df, aes(x, y, color = density)) + # 散布図
  geom_point(data = pi_truth_df, aes(x, y), shape = 3, size = 5) + # piの真の値
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # プロットの色
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  labs(title = "Dirichlet Distribution", 
       subtitle = paste0("N=", N, ", alpha=(", paste(alpha_k_hat, collapse = ", "), ")"), 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = ""))) # ラベル


# 3.2.2 予測分布 ---------------------------------------------------------------


# 予測分布のパラメータを計算
pi_k_hat <- alpha_k_hat / sum(alpha_k_hat)
#pi_k_hat <- (apply(s_nk, 2, sum) + alpha_k) / sum(apply(s_nk, 2, sum) + alpha_k)

# 作図用のsの値
s_sk <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3)

# 予測分布を計算
predict_df <- tibble(
  k = seq(1, 3),  # 作図用の値
  prob = apply(pi_k_hat^s_sk, 1, prod) # 確率
)


# 作図
ggplot(predict_df, aes(k, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  labs(title = "Categorical Distribution", 
       subtitle = paste0("N=", N, ", pi_hat=(", paste(round(pi_k_hat, 2), collapse = ", "), ")")) # ラベル


# 3.2.2 gif ---------------------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメーターの初期値を指定
# 観測モデルのパラメータ
pi_k_truth <- c(0.3, 0.5, 0.2)

# 事前分布のパラメータ
alpha_k <- c(2, 2, 2)

# 試行回数
N <- 50


# 作図用のpiの値
pi <- tibble(
  pi_1 = rep(rep(seq(0, 1, by = 0.025), times = 41), times = 41), 
  pi_2 = rep(rep(seq(0, 1, by = 0.025), each = 41), times = 41), 
  pi_3 = rep(seq(0, 1, by = 0.025), each = 1681)
)

# 正規化
pi <- pi / apply(pi, 1, sum)

# 重複した組み合わせを除去(ハイスぺ機なら不要…)
pi <- pi %>% 
  mutate(pi_1 = round(pi_1, 3), pi_2 = round(pi_2, 3), pi_3 = round(pi_3, 3)) %>% 
  count(pi_1, pi_2, pi_3) %>% 
  select(-n) %>% 
  as.matrix()


# 事前分布を計算
posterior_df <- tibble(
  x = pi[, 2] + (pi[, 3] / 2),  # 三角座標への変換
  y = sqrt(3) * (pi[, 3] / 2),  # 三角座標への変換
  C_D = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)),  # 正規化項(対数)
  density = exp(C_D + apply((alpha_k - 1) * log(t(pi)), 2, sum)), # 確率密度
  N = 0 # 試行回数
)


# 初期値による予測分布のパラメーターを計算
pi_k_hat <- alpha_k / sum(alpha_k)

# 作図用のsの値
s_sk <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3)

# 初期値による予測分布を計算
predict_df <- tibble(
  k = seq(1, 3),  # 作図用の値
  prob = apply(pi_k_hat^s_sk, 1, prod),  # 確率
  N = 0 # 試行回数
)


# パラメーターを推定
s_nk <- matrix(0, nrow = N, ncol = 3) # 受け皿
for(n in 1:N){
  
  # カテゴリ分布に従うデータを生成
  s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = pi_k_truth) %>% 
    as.vector()
  
  # ハイパーパラメータを更新
  alpha_k <- s_nk[n, ] + alpha_k
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    x = pi[, 2] + (pi[, 3] / 2),  # 三角座標への変換
    y = sqrt(3) * (pi[, 3] / 2),  # 三角座標への変換
    C_D = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)),  # 正規化項(対数)
    density = exp(C_D + apply((alpha_k - 1) * log(t(pi)), 2, sum)), # 確率密度
    N = n # 試行回数
  )
  
  
  # 予測分布のパラメーターを計算
  pi_k_hat <- alpha_k / sum(alpha_k)
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    k = seq(1, 3),  # 作図用の値
    prob = apply(pi_k_hat^s_sk, 1, prod),  # 確率
    N = n # 試行回数
  )
  
  # 結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
apply(s_nk, 2, sum)


# piの真の値のプロット用データフレームを作成
pi_truth_df <- tibble(
  x = pi_k_truth[2] + (pi_k_truth[3] / 2),  # 三角座標への変換
  y = sqrt(3) * (pi_k_truth[3] / 2),  # 三角座標への変換
  N = seq(0, N)
)


## 事後分布
# 作図
posterior_graph <- ggplot() + 
  geom_point(data = posterior_df, aes(x, y, color = density)) + # 散布図
  geom_point(data = pi_truth_df, aes(x, y), shape = 3, size = 5) + # piの真の値
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # プロットの色
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  transition_manual(N) + # フレーム
  labs(title = "Dirichlet Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = ""))) # ラベル

# 描画
animate(posterior_graph)


## 予測分布
# 作図
predict_graph <- ggplot(predict_df, aes(k, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  transition_manual(N) + # フレーム
  labs(title = "Categorical Distribution", 
       subtitle = "N= {current_frame}") # ラベル

# 描画
animate(predict_graph)


