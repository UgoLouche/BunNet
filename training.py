import numpy as np

# put some flags here, used by deepSimTrain
train_discr = True
train_gen = True

# Define the training procedure following :
# author       = "A. Dosovitskiy and T. Brox",
# title        = "Generating Images with Perceptual Similarity Metrics based on Deep Networks",
# booktitle    = "Advances in Neural Information Processing Systems (NIPS)",
# year         = "2016",
# source       = https://lmb.informatik.uni-freiburg.de/Publications/2016/DB16c/
def deepSimTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    global train_discr
    global train_gen
    # pool a batch of sampless
    idX = np.random.randint(0, x_train.shape[0], batch_size)
    x_gan = x_train[idX]
    y_gan = np.ones((len(idX)))
    h1_gan = h1_train[idX]

    # run the discriminator on real data
    if train_discr:
        discr_real_loss = disc_model.train_on_batch(x_train[idX], np.zeros([batch_size]))
    else:
        discr_real_loss = disc_model.test_on_batch(x_train[idX], np.zeros([batch_size]))

    # run the discriminator on generated data
    fake  = gan_model.predict(x_train[idX])[0]
    if train_discr:
        discr_fake_loss  = disc_model.train_on_batch(fake, np.ones([batch_size]))
    else:
        discr_fake_loss  = disc_model.test_on_batch(fake, np.ones([batch_size]))

    # run the discriminator on generated data with opposite labels, to get the gradient for the generator
    discr_fake_for_generator_loss = disc_model.test_on_batch(fake, np.zeros([batch_size]))
    discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss

    # finally, run the generator on batch data
    if train_gen:
        gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])
    else:
        gan_loss = gan_model.test_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    if discr_loss_ratio < 1e-1 and train_discr:
        train_discr = False
        train_gen = True
        print("<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
    if discr_loss_ratio > 5e-1 and not train_discr:
        train_discr = True
        train_gen = True
        print(" <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
    if discr_loss_ratio > 1e1 and train_gen:
        train_gen = False
        train_discr = True
        print(" <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))

    return (discr_fake_loss, gan_loss)


## Definbe custom GAN training procedure based on http://www.nada.kth.se/~ann/exjobb/hesam_pakdaman.pdf
#Do 5 disc iterations for one gan iteration. Except for the 500 first epoch and every 500 subsequent epochs
#where disc is trained 100 times
#Based on implementation found in https://github.com/hesampakdaman/ppgn-disc/blob/master/src/vanilla.py
def customGANTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    disc_train = 100 if (epochID < 25 or epochID % 500 == 0) else 5
    disc_loss, disc_pred = [], []
    #train disc
    for i in range(disc_train):
        idX = np.random.randint(0, x_train.shape[0], batch_size)

        valid = x_train[idX]
        fake  = gan_model.predict(x_train[idX])[0]
        x_disc = np.concatenate((valid, fake), axis=0)
        y_disc = np.concatenate((np.ones((batch_size)), np.zeros((batch_size))))

        disc_loss.append(disc_model.train_on_batch(x_disc, y_disc))
        disc_pred.append(100 * np.mean(np.round(disc_model.predict(x_disc)) == y_disc))
        #disc_pred.append(disc_model.predict(x_disc))

    #print('GAN/Disc {:d} train, test {:.2f} +/- {:.2f}'.format(disc_train, np.mean(disc_pred), np.std(disc_pred)))
    #train gen
    x_gan = x_train[idX][-1:]
    y_gan = np.ones((1)) #((len(idX)))
    h1_gan = h1_train[idX][-1:]

    gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    return (np.mean(disc_loss), gan_loss)

def simpleGANTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    #train disc
    idX = np.random.randint(0, x_train.shape[0], batch_size)

    valid = x_train[idX]
    fake  = gan_model.predict(x_train[idX])[0]
    x_disc = np.concatenate((valid, fake), axis=0)
    y_disc = np.concatenate((np.zeros((batch_size)), np.ones((batch_size))))

    disc_loss = disc_model.train_on_batch(x_disc, y_disc)
    #disc_pred.append(100 * np.mean(disc_model.predict(x_disc) == y_disc))
    disc_pred = disc_model.predict(x_disc).T[0]
    print('GAN/Disc {:.2f}'.format(100 * np.mean(np.round(disc_pred) == y_disc)))
    #train gen
    x_gan = x_train[idX]
    y_gan = np.ones((len(idX)))
    h1_gan = h1_train[idX]

    gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    return (disc_loss, gan_loss)
