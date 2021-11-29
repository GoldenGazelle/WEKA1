import java.io.BufferedReader;
import java.io.*;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class WekaLab1 {
    public static void main (String []args) throws IOException {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("data/bmw-training.arff"));
            Instances train = new Instances(reader);
            train.setClassIndex(train.numAttributes() - 1);
            BufferedReader reader2 = new BufferedReader(new FileReader("data/bmw-testing.arff"));
            Instances test = new Instances(reader2);
            test.setClassIndex(test.numAttributes() - 1);
            if(!train.equalHeaders(test))
                throw new IllegalArgumentException("Train and test set are not compatible!");

            J48 cls = new J48();
            cls.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            System.out.println("------------Classification without test set------------");
            eval.evaluateModel(cls, train);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            J48 cls1 = new J48();
            cls1.buildClassifier(train);
            Evaluation eval1 = new Evaluation(train);
            System.out.println("------------Classification with test set------------");
            eval1.evaluateModel(cls1, test);
            System.out.println(eval1.toSummaryString("\nResults\n======\n", false));

//            Evaluation eval = new Evaluation(train);
//            eval.crossValidateModel(cls, train, 10, new Random(1));
//            System.out.println(eval.toSummaryString("\nResults\n======\n", false));


//            System.out.println("#-actual-predicted-error-distribution");
////            for (int i = 0; i < train.numInstances(); i++) {
////                double pred = cls.classifyInstance(train.instance(i));
////                System.out.println("->");
////                System.out.println(train.classAttribute().value((int)pred));
////                System.out.println("\n Weka ended");
////            }
//            for (int i = 0; i < train.numInstances(); i++) {
//                double[] arr = cls.distributionForInstance(test.instance(i));
//                System.out.println("->");
//                System.out.println(arr[0] + ", " + arr[1]);
//                System.out.println("\n Weka ended");
//            }

                
        }
        catch (Exception e){
            System.out.println("Weka Error: " + e.getMessage());
        }
    }
}
