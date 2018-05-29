def init_weight(layer, std=0.02):
    layer.weight.data.normal_(0.0, std)


def init_weights(module):
    for c in module.children():
        init_weight(c)

