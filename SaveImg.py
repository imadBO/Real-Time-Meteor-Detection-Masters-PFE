import cv2

class SaveImg:

    @staticmethod
    def saveJPEG(img, name):
        """
        Save JPEG.
        
        Args:
            img: Opencv mat image to save.
            name: Path and name of the file to save.
        
        Returns:
            Success status to save image.
        """
        return cv2.imwrite(name+".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    @staticmethod
    def saveBMP(img, name):
        """
        Save BMP.
        
        Args:
            img: Opencv mat image to save.
            name: Path and name of the file to save.
        
        Returns:
            Success status to save image.
        """
        return cv2.imwrite(name+".bmp", img)

    @staticmethod
    def savePNG(img, name):
        """
        Save PNG.
        
        Args:
            img: Opencv mat image to save.
            name: Path and name of the file to save.
        
        Returns:
            Success status to save image.
        """
        return cv2.imwrite(name+".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 98])