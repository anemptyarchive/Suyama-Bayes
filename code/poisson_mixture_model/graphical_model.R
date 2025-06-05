
# chapter 4.3
# ポアソン混合モデル

# グラフィカルモデル表現 ---------------------------------------------------------------

# 利用パッケージ
library(DiagrammeR)
library(DiagrammeRsvg)


# グラフィカルモデルの作図 ------------------------------------------------------------

# ポアソン混合モデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'Poisson mixture model'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    # ハイパーパラメータ
    a [label = 'a']
    b [label = 'b']
    alpha [label = <<B>&alpha;</B>>]
    
    # パラメータ
    subgraph cluster_k{
      label    = 'K'
      fontsize = 14
      
      lambda [label = '&lambda;@_{k}']
    }
  
    pi [label = <<B>&pi;</B>>]
    
    
    # 観測・潜在変数
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      s [label = <<B>s</B>@_{n}>]
      x [label = 'x@_{n}', style = filled, filledcolor = gray]
      
      {rank = same; s, x}
    }
    
    {a, b} -> lambda -> x;
    alpha -> pi -> s -> x;
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 1500) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/Poisson_mixture_model.png", dpi = 100) # pngファイルに変換


