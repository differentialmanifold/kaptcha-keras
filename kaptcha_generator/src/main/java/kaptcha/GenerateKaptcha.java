package kaptcha;

import com.google.code.kaptcha.impl.DefaultKaptcha;
import com.google.code.kaptcha.util.Config;
import org.apache.commons.io.FileUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Properties;

public class GenerateKaptcha {

    private DefaultKaptcha getKaptchaBean(){
        DefaultKaptcha defaultKaptcha=new DefaultKaptcha();
        Properties properties=new Properties();
        properties.setProperty("kaptcha.border", "yes");
        properties.setProperty("kaptcha.border.color", "105,179,90");
        properties.setProperty("kaptcha.textproducer.font.color", "blue");
        properties.setProperty("kaptcha.image.width", "125");
        properties.setProperty("kaptcha.image.height", "45");
        properties.setProperty("kaptcha.session.key", "code");
        properties.setProperty("kaptcha.textproducer.char.length", "4");
        properties.setProperty("kaptcha.textproducer.font.names", "宋体,楷体,微软雅黑");

        Config config=new Config(properties);
        defaultKaptcha.setConfig(config);
        return defaultKaptcha;
    }

    private void createDirectory(String directoryPath) throws IOException {
        File directory = new File(directoryPath);

        if (directory.exists()) {
            FileUtils.deleteDirectory(directory);
        }

        directory.mkdirs();
    }

    public static void main(String[] args) throws IOException {

        Integer imageNum;
        String dataDir = "/tmp/data";

        if (args.length == 1) {
            imageNum = Integer.parseInt(args[0]);
        } else if (args.length == 2) {
            imageNum = Integer.parseInt(args[0]);
            dataDir = args[1];
        } else {
            throw new IllegalArgumentException("args len is " +  args.length);
        }
        GenerateKaptcha generateKaptcha = new GenerateKaptcha();


        generateKaptcha.createDirectory(dataDir);

        DefaultKaptcha captchaProducer = generateKaptcha.getKaptchaBean();

        String passPath = dataDir + "/pass.txt";
        String imgPath = null;

        for (int i = 0; i<imageNum; i++ ) {

            if (i % 1000 == 0) {
                System.out.println("i is " + i);
            }
            String capText = captchaProducer.createText();

            BufferedImage bi = captchaProducer.createImage(capText);

            imgPath = dataDir + "/" + i + ".jpg";
            FileOutputStream fileOutputStream = null;
            try {
                fileOutputStream = new FileOutputStream(imgPath, false);
                ImageIO.write(bi, "jpeg", fileOutputStream);

                OutputStream os = new FileOutputStream(new File(passPath), true);
                String writeStr = capText + " ";
                os.write(writeStr.getBytes(), 0, writeStr.length());
                os.close();
            } catch (Exception e) {
                System.err.println(e);
            } finally {
                try {
                    if(fileOutputStream!=null){
                        fileOutputStream.close();
                    }
                } catch (IOException e) {
                    System.err.println(e);
                }
            }
        }


    }
}
