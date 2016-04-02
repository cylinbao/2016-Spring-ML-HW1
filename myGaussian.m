function Y = myGaussian(x1, x2, mean_x1, mean_x2, var_x1, var_x2)

Y = exp(-((x1-mean_x1)^2/(2*var_x1)) - ((x2-mean_x2)^2/(2*var_x2)));

endfunction
