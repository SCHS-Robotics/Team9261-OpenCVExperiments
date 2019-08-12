import java.util.function.Function;

public class VelocitySmoother {

    private double aMax;
    private long lastGenerationTime;
    private double currentTarget;
    private double profileStartVelocity;
    private double t_critical;
    private boolean transitioning;
    private Function<Double,Double> velocityProfile;
    private Function<Double,Double> accelerationProfile;
    private BiFunction<Double,Double,Double> jerkProfile;

    private double dP,t0,xShift,yShift;

    //private SmoothstepFunction smoother;


    public VelocitySmoother(double accelMax) {
        aMax = Math.abs(accelMax);
        velocityProfile = (Double time)-> (dP*(3*Math.pow((time)/t0,2)-2*Math.pow((time)/t0,3))+yShift);
        accelerationProfile = (Double time) -> (((6*dP)/Math.pow(t0,2))*(time)-((6*dP)/Math.pow(t0,3))*Math.pow(time,2));
        jerkProfile = (Double shift, Double time) -> ((6*dP)/Math.pow(t0,2) - ((12*dP)/Math.pow(t0,3))*(time-shift));
        dP = 0;
        t0 = 1;
        xShift = 0;
        yShift = 0;
        lastGenerationTime = 0;
        profileStartVelocity = 0;
        t_critical = 0;
        transitioning = false;
    }

    public void setInitialVelocity(double initialVelocity) {
        profileStartVelocity = initialVelocity;
    }

    public double getVelocity() {
        System.out.println("t internal: " + (System.currentTimeMillis()-lastGenerationTime)/1000.0);
        if(transitioning && System.currentTimeMillis() >= t_critical) {
            transitioning = false;
            setInitialVelocity(velocityProfile.apply(0.0));
            updateProfiles(currentTarget);
        }
        return (System.currentTimeMillis()-lastGenerationTime)/1000.0 < t0 ? velocityProfile.apply((System.currentTimeMillis()-lastGenerationTime)/1000.0) : currentTarget;
    }

    public void updateProfiles(double target_velocity) {

        if(lastGenerationTime == 0) {
            lastGenerationTime = System.currentTimeMillis();
            currentTarget = target_velocity;
        }

        double t = (System.currentTimeMillis()-lastGenerationTime)/1000.0;



        if(target_velocity == currentTarget) {
            double dP = target_velocity-profileStartVelocity;
            double t0 = (3.0*dP)/(2.0*sgn(dP)*aMax);

            this.yShift = profileStartVelocity;
            this.t0 = t0;
            this.dP = dP;
            lastGenerationTime = System.currentTimeMillis();

        }

        else if(true) {
            currentTarget = target_velocity;
            profileStartVelocity = velocityProfile.apply(t);

            double currentAcceleration = accelerationProfile.apply(t);

            double dP = target_velocity - profileStartVelocity;
            double t0 = (3.0 * dP) / (2.0 * sgn(dP) * aMax);

            double k1 = 2 * t - t0;
            double k2 = t0 * t - Math.pow(t, 2) - currentAcceleration * (Math.pow(t0, 3) / (6 * dP));

            double shift1 = (k1 - Math.sqrt(Math.pow(k1, 2) + 4 * k2)) / 2.0;
            double shift2 = (k1 + Math.sqrt(Math.pow(k1, 2) + 4 * k2)) / 2.0;

            this.t0 = t0;
            this.dP = dP;

            double jerk1 = jerkProfile.apply(shift1, t);
            double jerk2 = jerkProfile.apply(shift2, t);

            if (sgn(jerk1) == sgn(jerk2) && jerk1 != 0 && jerk2 != 0) {
                System.out.println("Bad things are happening");
            }

            xShift = sgn(jerk1) == sgn(dP) ? shift1 : shift2;

            yShift = 0;
            yShift = profileStartVelocity - velocityProfile.apply(t - xShift);
            lastGenerationTime = System.currentTimeMillis() + (long) (1000 * (xShift - t));
            t_critical = lastGenerationTime;

            if (sgn(dP) == sgn(velocityProfile.apply(t - xShift)) && velocityProfile.apply(0.5*t0-xShift) == currentTarget) {
                transitioning = false;
            }

            else if(sgn(dP) == sgn(velocityProfile.apply(t - xShift))) {
                transitioning = true;
                lastGenerationTime = System.currentTimeMillis() + (long) (1000 * (0.5*t0));
                t_critical = lastGenerationTime;
            }

            else {
                transitioning = true;
            }
        }
        else {
            currentTarget = target_velocity;
        }

        System.out.println("dP: "+dP);
        System.out.println("t0: "+t0);
        System.out.println("t: "+t);
        System.out.println("xShift: "+xShift);
        System.out.println("yShift: "+yShift);
        System.out.println("critical time: "+ (System.currentTimeMillis()-t_critical)/1000.0);
        System.out.println("transitioning: "+transitioning);

    }

    private int sgn(double num) {
        return num < 0 ? -1 : 1;
    }
}
