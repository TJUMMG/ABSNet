
from codec_drn import DINetMTL


def save_encoder(model, log_dir, cur_epoch, args):
    import time
    dday = time.strftime("%Y%m%d",time.localtime())
    encoder = model.get_layer(index=1) #[input,encoder,decoder]
    # encName = checkpointdir
    encoder.save(log_dir+'/EncOfDINet.'+ 'DATE{}-ep{:02}-{}-lr={}-bs={}'.format(dday,cur_epoch,args.database,str(args.lr),str(args.batch_size))+'.pkl')

def _train_sal_drn(cur_epoch):

    def _scheduler(epoch):
        if epoch%2==0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            if lr > 1e-7:
                K.set_value(model.optimizer.lr, lr*.1)
                print("lr changed to {}".format(lr*.1))
        return K.get_value(model.optimizer.lr)
    
    parser = ArgumentParser(description='DINet')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    # parser.add_argument('--weight_decay', type=float, default=0.0,
    #                     help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='Salicon', type=str,
                        help='database name (default: Salicon)')
    parser.add_argument('--phase', default='train', type=str,
                        help='phase (default: train')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained weight (default: False')       
    parser.add_argument('--trainbn', default= False,
                        help='train bn layer? (default: False')               

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    # print('model: ' + args.model)
    config.update(config[args.database])
    # config.update(config[args.model])


    b_s = args.batch_size 
    trainbn = args.trainbn
    shape_r =  config['shape_r']  
    shape_c =  config['shape_c'] 
    imgs_train_path = config['imgs_train_path'] 
    maps_train_path = config['maps_train_path'] 
    imgs_val_path = config['imgs_val_path'] 
    maps_val_path = config['maps_val_path']
    
    train_images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]    
    train_images.sort()
    train_maps.sort()
    val_images.sort()
    val_maps.sort()
    # print val_images
    if args.phase == 'train':
        g = tf.Graph()
        sess = tf.Session(graph=g)
        with sess.as_default():
            with sess.graph.as_default():
                import time
                dday = time.strftime("%Y%m%d",time.localtime())
                log_dir = 'logs/'+dday+'.stl.drn'
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                checkpointdir= log_dir+'/DINet.TVdist.'+ 'DATE{}-ep{:02}-{}-lr={}-bs={}'.format(dday,cur_epoch,args.database,str(args.lr),str(args.batch_size))
                print (checkpointdir)

                #-----------------------------------------#
                model = DINetSal(img_cols=shape_c, img_rows=shape_r,train_bn=trainbn,weight_file=None)
                # model.summary()
                if args.pretrained==True and cur_epoch==1:
                    print("Load weights DINet Sal")
                    # weight_file = 'models/'+ 'DINet.TVdist.EXP0-Salicon-lr=0.0001-bs=10.05-0.2638-0.2038.pkl'
                    # weight_file = 'logs/20200501.drn/DINet.TVdist.DATE20200501-Salicon-lr=0.0001-bs=10.04-0.1987-0.1473.pkl'
                    # model.load_weights(weight_file)
                    print("Load weight DINet SharedEncoder from ep#01") #load in codec_drn

                from glob import glob
                if cur_epoch <= 1:
                    enc_wf_list=glob(log_dir+'/EncOfDINet.*')
                    # sal_weight_file=None
                    sal_weight_file='logs/20200514.stl.drn/DINet.TVdist.DATE20200514-Salicon-lr=0.0001-bs=10.03-0.1982-0.1567.pkl'
                    enc_weight_file=enc_wf_list[-1]
                    model.get_layer(index=1).load_weights(enc_weight_file)
                    print("load encoder weights from {}".format(enc_weight_file))
                    model.load_weights(sal_weight_file)
                else:
                    ###use aes trained encoder
                    enc_wf_list = glob(log_dir+'/EncOfDINet.*')
                    sal_wf_list = glob(log_dir+'/DINet.TVdist.DATE*')
                    # enc_weight_file=log_dir+'/EncOfDINet.TVdist.'+ 'DATE{}-ep{}-{}-lr={}-bs={}'.format(dday,str(cur_epoch-1),args.database,str(args.lr),str(args.batch_size))+'.pkl'
                    # sal_weight_file=log_dir+'/DINet.TVdist.'+ 'DATE{}-ep{}-{}-lr={}-bs={}'.format(dday,str(cur_epoch-1),args.database,str(args.lr),str(args.batch_size))+'.pkl'
                    enc_weight_file=enc_wf_list[-1]
                    sal_weight_file=sal_wf_list[-1]
                    model.load_weights(sal_weight_file)
                    model.get_layer(index=1).load_weights(enc_weight_file)
                    print("load encoder weights from {}".format(enc_weight_file))

                model.compile(Adam(lr=args.lr), loss= [TVdist])
                # import keras
                # model.compile(keras.optimizers.SGD(lr=1e-5), loss= [TVdist])
                lr_decay = LearningRateScheduler(_scheduler)
                #-----------------------------------------#

                nb_imgs_train =  len(train_images) 
                nb_imgs_val =  len(val_images) 
                train_index = range(nb_imgs_train)
                val_index = range(nb_imgs_val)
                print (nb_imgs_train,nb_imgs_val)
                print("Training DINet Sal")
                
                train_generator = DataGenerator(train_index,train_images,train_maps,config, b_s, shuffle=True, mirror= False)    
                val_generator = DataGenerator(val_index,val_images,val_maps,config, 1, shuffle=False, mirror= False)
                model.fit_generator(generator=train_generator,
                                    epochs=15+cur_epoch,
                                    steps_per_epoch= int(nb_imgs_train // b_s),
                                    validation_data=val_generator, 
                                    validation_steps= int(nb_imgs_val // 1),
                                    callbacks=[EarlyStopping(patience=10),
                                            ModelCheckpoint(checkpointdir+'.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl', 
                                            save_best_only=True), ###
                                            lr_decay],
                                    initial_epoch=cur_epoch)
                save_encoder(model,log_dir,cur_epoch,args)
        
        tf.keras.backend.clear_session()


def _train_aes_drn(cur_epoch):
    def _scheduler(epoch):
        if epoch%2==0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            if lr > 1e-7:
                K.set_value(model.optimizer.lr, lr*.1)
                print("lr changed to {}".format(lr*.1))
        return K.get_value(model.optimizer.lr)

    parser = ArgumentParser(description='DINet')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    # parser.add_argument('--weight_decay', type=float, default=0.0,
    #                     help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='AVA', type=str,
                        help='database name (default: AVA)')
    parser.add_argument('--phase', default='train', type=str,
                        help='phase (default: train')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained weight (default: False')       
    parser.add_argument('--trainbn', default= False,
                        help='train bn layer? (default: False')
    parser.add_argument('--enc_weight_file', default= None, type=str, 
                        help='encoder weight file? (default: None')               

    args = parser.parse_args()

    from aes_dataloader import get_batched_dataset,get_batched_dataset2
    nb_imgs_train = 10000
    from aes_dataloader import BATCH_SIZE
    b_s = BATCH_SIZE
    ava_train_tfrecs_path = "/media/ubuntu/60E91B872FDF4622/ljc/data/ava_tfrecords/resized_tfrecords10000/"
    ava_val_tfrecs_path = '/media/ubuntu/074E446A247F62BF/lvjc/data/ava_tfrecords/tfrecords10000/test'
    import time
    dday = time.strftime("%Y%m%d",time.localtime())
    log_dir = 'logs/'+dday+'.stl.drn'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    checkpointdir= log_dir+'/DINet.EMDdist.'+ '{}-lr={}-bs={}'.format(
        "AVA",#args.database,
        0.0001,#str(args.lr),
        b_s,)#str(args.batch_size))
    logging = TensorBoard(log_dir=log_dir)
    print(checkpointdir)

    #-----------------------------------------#
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with sess.as_default():
        with sess.graph.as_default():
            # wf ='models/fromAuthors/EncOfDINet.TVdist.EXP0-Salicon-lr=0.0001-bs=10.05-0.2638-0.2038.h5'
            wf = 'logs/20200514.stl.drn/EncOfDINet.TVdist.DATE20200514-Salicon-lr=0.0001-bs=10.01-0.2134-0.2180.pkl'
            model =DINetAes(weight_file=wf)
            # model =DINetAes(weight_file=args.enc_weight_file)#img_cols=shape_c, img_rows=shape_r,train_bn= args.trainbn,weight_file) 
            # model.summary()
            if args.pretrained==True and cur_epoch==0:
                print("Load weights DINet Aes")
                #load encoder weights in 'DINetAes'

            if cur_epoch <= 1:
                enc_weight_file=None
                aes_weight_file=None
            else:
                from glob import glob
                ###use sal trained encoder 
                enc_wf_list = glob(log_dir+'/EncOfDINet.DATE*')
                aes_wf_list = glob(log_dir+'/DINet.EMDdist.*')
                # enc_weight_file=log_dir+'/EncOfDINet.TVdist.'+ 'DATE{}-ep{}-{}-lr={}-bs={}'.format(dday,str(cur_epoch-1),args.database,str(args.lr),str(args.batch_size))+'.pkl'
                # sal_weight_file=log_dir+'/DINet.TVdist.'+ 'DATE{}-ep{}-{}-lr={}-bs={}'.format(dday,str(cur_epoch-1),args.database,str(args.lr),str(args.batch_size))+'.pkl'
                # import pdb; pdb.set_trace()
                enc_weight_file=enc_wf_list[-1]
                aes_weight_file=aes_wf_list[-1]
                model.load_weights(aes_weight_file)
                model.get_layer(index=1).load_weights(enc_weight_file)
                print("load encoder weights from {}".format(enc_weight_file))
            
            model.compile(Adam(lr=1e-5), loss= [emd_loss])
            lr_decay = LearningRateScheduler(_scheduler)
            # model.summary()
            # import pdb; pdb.set_trace()
            #-----------------------------------------#

            #
            from utilities import config_keras_backend,clear_keras_session
            config_keras_backend()
            from glob import glob
            def get_itr(ava_tfrecs_path,is_training):
                filenames = glob(ava_tfrecs_path+'*.tfrecords*')
                dataset = get_batched_dataset(filenames,n_ep=10,is_training=is_training)
                itr = dataset.make_one_shot_iterator()
                features_itr, labels_itr = itr.get_next()
                return features_itr,labels_itr

            train_ftitr, train_labitr = get_itr(ava_train_tfrecs_path,is_training=True)
            val_ftitr, val_labitr = get_itr(ava_val_tfrecs_path,is_training=False)
            

            print("Training DINet Aes")
            model.fit(x=train_ftitr, y=[train_labitr],
                    epochs=15+cur_epoch,
                    steps_per_epoch=nb_imgs_train//b_s,
                    validation_data=(val_ftitr, val_labitr),
                    validation_steps=1000,
                    callbacks=[EarlyStopping(patience=10),
                    ModelCheckpoint(checkpointdir+'.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl', 
                    save_best_only=True), ###
                    lr_decay],
                    initial_epoch=cur_epoch)
            save_encoder(model,log_dir,cur_epoch,args)
    tf.keras.backend.clear_session()
    


if  __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import tensorflow as tf
    for i in range(0,20):
        print("Epoch #{}-----------\n".format(i))
        if i%2==0:
            from train_aes_drn import *
            _train_aes_drn(cur_epoch=i)
        if i%2==1:
            from main_sal_drn import *
            _train_sal_drn(cur_epoch=i)
        print("Epoch #{} done-----------\n".format(i))
    
    # from train_aes_drn import *
    # _train_aes_drn(cur_epoch=0)

    # from main_sal_drn import *
    # _train_sal_drn(cur_epoch=0)#train sal to the best
