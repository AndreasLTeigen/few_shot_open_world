from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    #palette = 10*color_list
    # palette = np.array(sns.color_palette(color_palette, num_classes))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def tile_scatter(x, colors, input_data):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)
    #####
    tx, ty = x[:, 0], x[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    plt.plot(tx, ty, '.')
    width = 1000
    height = 1000
    max_dim = 100
    full_image = Image.new('RGB', (width, height), (255, 255, 255))
    images = input_data.iloc[:, 0]

    box_color = 10*box_color_list

    for img, x, y, c in tqdm(zip(images, tx, ty, colors)):
        tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((max(1, int(tile.width / rs)), max(1, int(tile.height / rs))), Image.ANTIALIAS)

        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], fill=None, outline=box_color[c], width=5)

        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)))

    #####
    plt.imshow(full_image)

    f = plt.figure(figsize=(16, 12))


    return f

class TSNEAlgo():
    def __init__(self,
                 k_way: int,
                 n_shot: int,
                 ):
        self.k_way=k_way
        self.n_shot=n_shot
    def tsne_fit(self, X, input_data, title='TSNE'):
        labels = []
        for i in self.k_way:
            for j in self.n_shot:
                labels.append(i)
        X = pd.DataFrame(X)
        x = StandardScaler().fit_transform(X)
        time_start = time()
        RS = 123
        tsne = TSNE(random_state=RS).fit_transform(x)

        print('-- TSNE DONE! Time elapsed: {} seconds'.format(time() - time_start))

        f_tsne, ax_tsne, _, _ = fashion_scatter(tsne, labels)
        ax_tsne.set_title(title, fontsize=10)
        #f_tsne.show()
        plt.show()

        ## -- drawing the full images on TSNE
        f_tsne2 = tile_scatter(tsne, labels, input_data)
        plt.title(title)
        #f_tsne2.show()
        plt.show()
        return
