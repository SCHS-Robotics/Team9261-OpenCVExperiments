public class VelocitySmootherTest {
    public static void main(String args[]) {
        VelocitySmoother velocitySmoother = new VelocitySmoother(1);
        System.out.println(velocitySmoother.getClass().getName());
        long startTime = System.currentTimeMillis();
        double currentVelocity = 0;
        double target = 0.5;
        boolean run = false;
        boolean run2 = false;
        boolean run3 = false;
        velocitySmoother.setInitialVelocity(currentVelocity);
        velocitySmoother.updateProfiles(target);
        while((System.currentTimeMillis()-startTime)/1000.0 < 0.8) {

            if((System.currentTimeMillis()-startTime)/1000.0 >= 0.3 && !run) {
                velocitySmoother.updateProfiles(0);
                run = true;
            }
            if((System.currentTimeMillis()-startTime)/1000.0 >= 0.34 && !run2) {
                velocitySmoother.updateProfiles(0.3);
                run2 = true;
            }
            if((System.currentTimeMillis()-startTime)/1000.0 >= 0.386 && !run3) {
                velocitySmoother.updateProfiles(1);
                run3 = true;
            }

            System.out.println("t: " + (System.currentTimeMillis()-startTime)/1000.0);
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
