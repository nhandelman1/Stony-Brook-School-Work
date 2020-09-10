x = [0:.01:20];
mu=2;
sd=3;

%find K = ceiling of maximum of gamma(x,3,1) / Normal(x,u,s)
%This maximum K ensures that gamma(x,3,1) <= K * Normal(x,u,s) 
func=@(x)0.5*(sd*(2*pi)^0.5)*(x^2)*exp(-x)*exp(0.5*(x-mu)^2/(2*sd*sd));
funcneg=@(x)-func(x);
K_ceil=ceil(func(fminsearch(funcneg,2))); %fminsearch returns argmin
K_floor=K_ceil-1;

%Graphically confirm that gamma(x,3,1) <= K * Normal(x,u,s)
%From the graph, the inequality is clearly true up to 3 sd.
%Greater than 3 sd, zoom in to see that the inequality holds. 
figure('Name','Normal and Gamma Plots')
plot(x,K_ceil*normpdf(x,mu,sd),x,gampdf(x,3,1),x,K_floor*normpdf(x,mu,sd),x,normpdf(x,mu,sd)) 
legend(strcat(num2str(K_ceil),'*Normal(',num2str(mu),',',num2str(sd),')'),...
    'Gamma(3,1)',...
    strcat(num2str(K_floor),'*Normal(',num2str(mu),',',num2str(sd),')'),...    
    strcat('Normal(',num2str(mu),',',num2str(sd),')'))

samples10k=draw_samples(10000,mu,sd,func,K_ceil);
samples100k=draw_samples(100000,mu,sd,func,K_ceil);

%In each of the following cases, the fitted distribution is close to 
%gamma(3,1) so the rejection sampling reasonably samples a gamma(3,1)
figure('Name','10K Samples')
histfit(samples10k,100,'gamma')
gp=gamfit(samples10k);
legend('Histogram',strcat('Gamma(',num2str(gp(1)),',',num2str(gp(2)),')'))

figure('Name','100K Samples')
histfit(samples100k,100,'gamma')
gp=gamfit(samples100k);
legend('Histogram',strcat('Gamma(',num2str(gp(1)),',',num2str(gp(2)),')'))

function samples = draw_samples(num_samples,mu,sd,func,K_ceil)
    samples=zeros(num_samples,1);
    
    for i = 1:num_samples
        n=normrnd(mu,sd);

        %gamma dist defined (0,inf) but normal dist defined (-inf,inf)
        %there are n < 0 such that func(n)/K_ceil > 1. n < 0 are not 
        %possible samples from a gamma dist
        if n > 0 && rand <= func(n)/K_ceil 
            samples(i)=n;
        end    
    end
    samples=samples(samples~=0);
end
