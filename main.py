"""Main Method"""
import argparse
from time import sleep

from config_manager import ConfigManager
from reader import ImageReader, VideoReader
from writer import ImageWriter, VideoWriter

from composition.image_composition import ImageComposition


def main(args):
    """_summary_

    :param args: _description_
    :type args: _type_
    :raises ValueError: _description_
    """
    config_manager = ConfigManager()
    config = config_manager.get_config(args.env_path)
    image_compositer =  ImageComposition(config)
    bbox = [579, 988, 1332, 3582]

    if config.input_type == 'image':
        fg_image_reader = ImageReader(config.foreground_image_path)
        bg_image_reader = ImageReader(config.background_image_path)
        image_writer = ImageWriter()

        fg_image = fg_image_reader.get_image()
        bg_image = bg_image_reader.get_image()

        final_image, final_mask = image_compositer.process_image(fg_image, bg_image, bbox)

        image_writer.write_image(final_image, config.output_path, 'final_image.jpg')
        image_writer.write_image(final_mask, config.output_path, 'final_mask.jpg')

    elif config.input_type == 'composite':
        composite_frame_reader = ImageReader(config.composite_frame_path)
        composite_mask_reader = ImageReader(config.composite_mask_path)
        bg_image_reader = ImageReader(config.background_image_path)
        image_writer = ImageWriter()

        composite_frame = composite_frame_reader.get_image()
        composite_mask = composite_mask_reader.get_image()
        bg_image = bg_image_reader.get_image()

        final_image, final_mask = image_compositer.process_composite(composite_frame, composite_mask, bg_image)

        image_writer.write_image(final_image, config.output_path, 'final_image.jpg')
        image_writer.write_image(final_mask, config.output_path, 'final_mask.jpg')

    elif config.input_type == 'video':
        # INITIALIZING VIDEO READERS
        fg_video_reader = VideoReader(config.foreground_video_path)
        bg_video_reader = VideoReader(config.background_video_path)
        bg_video_prop = bg_video_reader.get_video_properties()

        # INITIALIZING VIDEO WRITER
        video_writer = VideoWriter('test_video.mp4', bg_video_prop)
        sleep(0.05)

        while True:

            ret_fg, fg_frame = fg_video_reader.read_frames()
            _, bg_frame = bg_video_reader.read_frames()

            if not ret_fg:
                break

            final_image, final_mask = image_compositer.process_image(fg_frame, bg_frame, bbox)
            video_writer.write_frame(final_image)

    else:
        raise ValueError("Input Type is not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str, default='config.env')
    main(parser.parse_args())
