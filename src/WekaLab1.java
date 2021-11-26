import java.awt.*;
import java.io.BufferedReader;
import java.io.*;
import weka.core.*;
import weka.core.Instances;
import weka.classifiers.trees.J48;

public class WekaLab1 {
    public static void main (String []args) throws IOException {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("data/bmw-training.arff"));
            Instances train = new Instances (reader);
            train.setClassIndex(train.numAttributes() - 1);
            BufferedReader reader2 = new BufferedReader(new FileReader("data/bmw-testing.arff"));
            Instances test = new Instances (reader2);
            test.setClassIndex(test.numAttributes() - 1);
            if(!train.equalHeaders(test))
                throw new IllegalArgumentException("Train and test set are not compatible!");
            J48 cls = new J48();
            cls.buildClassifier(train);
            System.out.println("#-actual-predicted-error-distribution");
            for (int i = 0; i < train.numInstances(); i++) {
                double pred = cls.classifyInstance(train.instance(i));
                System.out.println("->");
                System.out.println(train.classAttribute().value((int)pred));
                System.out.println("\n Weka ended");
            }
        }
        catch (Exception e){
            System.out.println("Weka Error" + e.getMessage());
        }
    }
}
