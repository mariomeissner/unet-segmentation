from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, UpSampling2D, concatenate
from kaf import KAF


def unet(
    input_size=(256, 256, 3),
    activation="relu",
    pretrained_weights=None,
    kaf_kernel="softplus",
    init_cfn=None,
    kaf_D=5,
):

    if activation == "kaf":
        activation = None

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation=activation, padding="same", kernel_initializer="he_normal")(inputs)
    if activation is None:
        conv1 = KAF(64, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv1)
    conv1 = Conv2D(64, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv1)
    if activation is None:
        conv1 = KAF(64, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation=activation, padding="same", kernel_initializer="he_normal")(pool1)
    if activation is None:
        conv2 = KAF(128, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv2)
    conv2 = Conv2D(128, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv2)
    if activation is None:
        conv2 = KAF(128, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation=activation, padding="same", kernel_initializer="he_normal")(pool2)
    if activation is None:
        conv3 = KAF(256, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv3)
    conv3 = Conv2D(256, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv3)
    if activation is None:
        conv3 = KAF(256, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=activation, padding="same", kernel_initializer="he_normal")(pool3)
    if activation is None:
        conv4 = KAF(512, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv4)
    conv4 = Conv2D(512, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv4)
    if activation is None:
        conv4 = KAF(512, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=activation, padding="same", kernel_initializer="he_normal")(pool4)
    if activation is None:
        conv5 = KAF(1024, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv5)
    conv5 = Conv2D(1024, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv5)
    if activation is None:
        conv5 = KAF(1024, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation=activation, padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(drop5)
    )
    if activation is None:
        up6 = KAF(512, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=activation, padding="same", kernel_initializer="he_normal")(merge6)
    if activation is None:
        conv6 = KAF(512, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv6)
    conv6 = Conv2D(512, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv6)
    if activation is None:
        conv6 = KAF(512, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv6)

    up7 = Conv2D(256, 2, activation=activation, padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv6)
    )
    if activation is None:
        up7 = KAF(256, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=activation, padding="same", kernel_initializer="he_normal")(merge7)
    if activation is None:
        conv7 = KAF(256, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv7)
    conv7 = Conv2D(256, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv7)
    if activation is None:
        conv7 = KAF(256, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv7)

    up8 = Conv2D(128, 2, activation=activation, padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv7)
    )
    if activation is None:
        up8 = KAF(128, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=activation, padding="same", kernel_initializer="he_normal")(merge8)
    if activation is None:
        conv8 = KAF(128, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv8)
    conv8 = Conv2D(128, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv8)
    if activation is None:
        conv8 = KAF(128, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv8)

    up9 = Conv2D(64, 2, activation=activation, padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv8)
    )
    if activation is None:
        up9 = KAF(64, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=activation, padding="same", kernel_initializer="he_normal")(merge9)
    if activation is None:
        conv9 = KAF(64, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv9)
    conv9 = Conv2D(64, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv9)
    if activation is None:
        conv9 = KAF(64, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv9)
    conv9 = Conv2D(3, 3, activation=activation, padding="same", kernel_initializer="he_normal")(conv9)
    if activation is None:
        conv9 = KAF(3, D=kaf_D, kernel=kaf_kernel, init_cfn=init_cfn, conv=True)(conv9)
    conv10 = Conv2D(3, 1, activation="softmax")(conv9)

    model = Model(input=inputs, output=conv10)

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
