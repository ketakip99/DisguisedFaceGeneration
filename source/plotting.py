import torch
from torch import device, set_grad_enabled
from torch.cuda import is_available
from torch.optim import Adam
import dataset
import losses
from tqdm import tqdm
import new_components as components


resume = True
batch_size = 4
epochs = 50
save_interval = 2
load_photo = True

train_path = '/home/asds/sk2df/source/'
val_path = '/home/asds/sk2df/source/'

criteria_mse = losses.MSE()
criteria_bce = losses.BCE()
criteria_mae = losses.MAE()

label_real = 1.0
label_fake = 0.0

# Loss files
discriminator_loss_file = "./discriminator_loss.txt"
generator_loss_file = "./generator_loss.txt"


if not torch.cuda.is_available():  
    with open(discriminator_loss_file, 'w') as f:
        f.write("Epoch,Discriminator Loss\n")

    with open(generator_loss_file, 'w') as f:
        f.write("Epoch,Generator Loss\n")


def main():
    modes = ['train']
    dataset_paths = {'train': train_path, 'val': val_path}
    device_name = device('cuda:0') if is_available() else device('cpu')
    print(f"Using device: {device_name}")

    spatial_channels = 1
    image_channels = 3
    gan_kwargs = {
        'generator': {
            'encoder_channels_list': [spatial_channels, 64, 128, 256],
            'residual_layers': 9,
            'decoder_channels_list': [256, 128, 64, image_channels]
        },
        'discriminator': {
            'disc_channels_list': [image_channels, 64, 128, 256, 512, 1],
            'pool': 0,
        }
    }

    gan = components.Cycle_GAN(**gan_kwargs)
    gan.to(device_name)

    if resume:
        gan.load_model(generator=True, discriminator=True)

    dataloaders = {
        x: dataset.dataloader(dataset_paths[x], batch_size=batch_size, shuffle=True, load_photo=load_photo, augmentation=False)
        for x in modes
    }

    optimizer_fm_generator = Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        running_loss = {x: { comp: 0.0 for comp in gan.components} 
                    for x in modes}
        discriminator_epoch_loss = 0.0
        generator_epoch_loss = 0.0
        batch_count = 0

        for x in modes:
            gan.train(x == 'train')
            for batch_index, (sketches, photos) in enumerate(tqdm(dataloaders[x], desc=f'Epoch {epoch}/{epochs}')):
                spatial_map = sketches.to(device_name)
                photos = photos.to(device_name)

                with set_grad_enabled(x == 'train'):
                    # Discriminator
                    optimizer_discriminator.zero_grad()
                    fakes = gan(spatial_map)
                    D_real = gan.discriminate(photos)
                    D_fake = gan.discriminate(fakes.detach())

                    D_loss_real = criteria_mse.compute(D_real, torch.ones_like(D_real))
                    D_loss_fake = criteria_mse.compute(D_fake, torch.zeros_like(D_fake))
                    D_loss = (D_loss_real + D_loss_fake) / 2
                    
                    running_loss[x][gan.components[1]] += D_loss.item() * len(sketches) / len(dataloaders[x].dataset)
                
                    print(f'Dis Losses: fake: {D_loss_fake}, real: {D_loss_real}, tot: {D_loss}')

                    if x == 'train':
                        D_loss.backward()
                        optimizer_discriminator.step()

                    discriminator_epoch_loss += D_loss.item()

                    # Generator
                    optimizer_fm_generator.zero_grad()
                    D_fake = gan.discriminate(fakes)
                    G_loss_MSE = criteria_mse.compute(D_fake, torch.ones_like(D_fake))
                    L1_loss = criteria_mae.compute(photos, fakes) * 10 
                    G_loss = G_loss_MSE + L1_loss
                    
                    running_loss[x][gan.components[0]] += G_loss.item() * len(sketches) / len(dataloaders[x].dataset)
                
                    print(f'Gen Losses: mse: {G_loss_MSE}, mae: {L1_loss}, tot: {G_loss}')

                    if x == 'train':
                        G_loss.backward()
                        optimizer_fm_generator.step()

                    generator_epoch_loss += G_loss.item()

                    batch_count += 1


        avg_D_loss = discriminator_epoch_loss / batch_count
        avg_G_loss = generator_epoch_loss / batch_count

        with open(discriminator_loss_file, 'a') as f:
            f.write(f"{epoch},{avg_D_loss:.6f}\n")

        with open(generator_loss_file, 'a') as f:
            f.write(f"{epoch},{avg_G_loss:.6f}\n")

        for key, losses in running_loss.items():
            loss_str = ', '.join([f'{comp}: {value:.6f}' for comp, value in losses.items()])
            print(f'Loss for {key} : {loss_str}')

        if epoch % save_interval == 0:
            gan.save_model()
            print(f"Model saved at epoch {epoch}")


if __name__ == '__main__':
    main()