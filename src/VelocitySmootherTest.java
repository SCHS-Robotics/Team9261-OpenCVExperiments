public class VelocitySmootherTest {
    public static void main(String args[]) {
        VelocitySmoother velocitySmoother = new VelocitySmoother(1);
        long startTime = System.currentTimeMillis();
        double currentVelocity = 0;
        double target = 0.5;
        boolean run = false;
        while((System.currentTimeMillis()-startTime)/1000.0 < 0.75) {

            if((System.currentTimeMillis()-startTime)/1000.0 >= 0.49 && !run) {
                currentVelocity = velocitySmoother.getVelocity();
                target = 0;
                run = true;
            }
            velocitySmoother.update(currentVelocity,target);
            System.out.println(velocitySmoother.getVelocity());
            System.out.println();
            try {
                Thread.sleep(10);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
