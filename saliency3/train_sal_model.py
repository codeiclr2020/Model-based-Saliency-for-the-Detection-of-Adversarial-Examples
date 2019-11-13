import torch
from sal.saliency_model import SaliencyModel, SaliencyLoss
from sal.datasets import my_dataset
from torchvision.models.resnet import resnet50
import matplotlib as plt
import sal.datasets
from sal.utils.pytorch_trainer import *
from sal.utils.pytorch_fixes import *
from sal.small import SimpleClassifier

def get_black_box_fn(model_zoo_model=resnet50, cuda=True, image_domain=(-1., 1.)):
    ''' You can try any model from the pytorch model zoo (torchvision.models)
        eg. VGG, inception, mobilenet, alexnet...
    '''
    black_box_model = model_zoo_model

    black_box_model.train(False)
    if cuda:
        black_box_model = torch.nn.DataParallel(black_box_model).cuda()

    def black_box_fn(_images):
        return black_box_model(adapt_to_image_domain(_images, image_domain))
    return black_box_fn

@TrainStepEvent()
@EveryNthEvent(4000)
def lr_step_phase1(s):
    print()
    print(GREEN_STR % 'Reducing lr by a factor of 10')
    for param_group in optim_phase1.param_groups:
        param_group['lr'] = param_group['lr'] / 10.


@ev_batch_to_images_labels
def ev_phase1(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = (_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, (NUM_CLASSES-1)).cuda()).long()*__fakes.long())%NUM_CLASSES
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    exists_loss = F.cross_entropy(_exists_logits, _is_real_label)
    loss = PT(loss=exists_loss)


@ev_batch_to_images_labels
def ev_phase2(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = PT(targets=(_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, (NUM_CLASSES-1)).cuda()).long()*__fakes.long())%NUM_CLASSES)
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    saliency_loss = saliency_loss_calc.get_loss(_images, _labels, _masks, _is_real_target=_is_real_label,  pt_store=PT)
    loss = PT(loss=saliency_loss)


@TimeEvent(period=5)
def phase2_visualise(s):
    pt = s.pt_store
    orig = auto_norm(pt['images'][0])
    mask = auto_norm(pt['masks'][0]*255, auto_normalize=False)
    preserved = auto_norm(pt['preserved'][0])
    destroyed = auto_norm(pt['destroyed'][0])
    print()
    print('Target (%s) = %s' % (GREEN_STR%'REAL' if pt['is_real_label'][0] else RED_STR%'FAKE!' , dts.CLASS_ID_TO_NAME[pt['targets'][0]]))
    final = np.concatenate((orig, mask, preserved, destroyed), axis=1)
    #plt.imshow(final)
    pycat.show(final)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_c', default='')
    parser.add_argument('--save_s', default='')
    parser.add_argument('--area_loss', type=float, default=8.0)
    parser.add_argument('--area_smooth', type=float, default=0.0)
    parser.add_argument('--preserver', type=float, default=0.3)
    parser.add_argument('--area_power', type=float, default=0.3)
    parser.add_argument('--load_model', type = bool, default=False)

    args = parser.parse_args()
    SAVE_CLASSIFIER = args.save_c
    SAVE_SAL_MODEL = args.save_s
    AREA_LOSS_COEF = args.area_loss
    SMOOTH_LOSS_COEF = args.area_smooth
    PRESERVER = args.preserver
    AREA_POWER = args.area_power

    # ---- config ----l
    # You can choose your own dataset and a black box classifier as long as they are compatible with the ones below.
    # The training code does not need to be changed and the default values should work well for high resolution ~300x300 real-world images.
    # By default we train on 224x224 resolution ImageNet images with a resnet50 black box classifier.
    #dts = imagenet_dataset
    dts = my_dataset

    NUM_CLASSES=2
    # ----------------

    train_dts = dts.get_train_dataset()
    val_dts = dts.get_val_dataset()

    model = SimpleClassifier(base_channels=64, num_classes=2)
    if args.load_model:
        model.restore(SAVE_CLASSIFIER)
    else:
        simple_img_classifier_train(
            model,
            dts,
            64, #batch size
            33, #epochs
            0.1, #lr
            15,
        )
        model.save(SAVE_CLASSIFIER)

    black_box_fn = get_black_box_fn(model_zoo_model=model)

    # Saliency Map
    saliency = SaliencyModel(model, 3, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=True,  num_classes=NUM_CLASSES)

    saliency_p = nn.DataParallel(saliency).cuda()
    saliency_loss_calc = SaliencyLoss(black_box_fn, smoothness_loss_coef=SMOOTH_LOSS_COEF,
                                      area_loss_coef = AREA_LOSS_COEF,
                                      preserver_loss_coef = PRESERVER,
                                      area_loss_power= AREA_POWER,
                                      num_classes=NUM_CLASSES) # model based saliency requires very small smoothness loss and therefore can produce very sharp masks
    optim_phase1 = torch_optim.Adam(saliency.selector_module.parameters(), 0.001, weight_decay=0.0001)
    optim_phase2 = torch_optim.Adam(saliency.get_trainable_parameters(), 0.001, weight_decay=0.0001)





    nt_phase1 = NiceTrainer(ev_phase1, dts.get_loader(train_dts, batch_size=128), optim_phase1,
                     val_dts=dts.get_loader(val_dts, batch_size=128),
                     modules=[saliency],
                     printable_vars=['loss', 'exists_accuracy'],
                     events=[lr_step_phase1,],
                     computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
    FAKE_PROB = .5
    for j in range(5):
        nt_phase1.train(steps = 8500)

    print(GREEN_STR % 'Finished phase 1 of training, waiting until the dataloading workers shut down...')

    nt_phase2 = NiceTrainer(ev_phase2, dts.get_loader(train_dts, batch_size=64), optim_phase2,
                     val_dts=dts.get_loader(val_dts, batch_size=64),
                     modules=[saliency],
                     printable_vars=['loss', 'exists_accuracy'],
                     events=[phase2_visualise,],
                     computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
    FAKE_PROB = .3
    for k in range(5):
        nt_phase2.train(steps = 3000)
    saliency.minimalistic_save(SAVE_SAL_MODEL)  # later to restore just use saliency.minimalistic_restore method.