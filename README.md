# generate_title

<p>Youtubeに使用したいタイトルを入力すると、意味は保ったまま表現の異なる文章を出力してくれる</p>
![image](https://user-images.githubusercontent.com/65533712/107841420-f6d15700-6dfd-11eb-81cf-b9c208b68e94.png)


<p>Google Colab上で制作</p>
https://colab.research.google.com/drive/1D_loNO3y9XxJrCRYEPjXNK8jEsrpKhFo?usp=sharing

# 使用した言語 & フレームワーク
<ul>
  <li>python</li>
  <li>pytorch</li>
</ul>

# 制作の流れ
<ol>
  <li>スクレイピングで、複数名Youtuberのタイトルを取得（約14,000件）</li>
  <li>分かち書きや単語IDなどの前処理</li>
  <li>VAEとLSTMを組み合わせたモデルの構築</li>
  <li>60エポックで学習（KL誤差は徐々に適応させる）</li>
  <li>予測、生成文章の表示</li>
</ol>

# 工夫した点
<ol>
  <li>Youtuber初心者が良いタイトルを生成できるように、データは再生回数が多い方のタイトルを使用した。</li>
  <li>Decoderだけで学習をしてしまう問題を防ぐため、Decoderに入力をする文章は30%ほど無情報ベクトルにした。</li>
  <li>VAEの損失関数の1つであるKL誤差が、すぐに収束してしまう問題を回避するため、KL誤差は徐々に適応させた。</li>
</ol>

# 今後の展望
<ol>
  <li>精度向上のため、データ数を増やしたり、再生回数や動画評価などの項目も学習時に加えてみる。</li>
  <li>楽しいタイトルや悲しいタイトルといった感情を考慮したものや、〇〇っぽいタイトルなど特定のラベルを指定したタイトル生成をしたい。</li>
</ol>
