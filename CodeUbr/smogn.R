suppressPackageStartupMessages(library(operators))
suppressPackageStartupMessages(library(class))
suppressPackageStartupMessages(library(fields))
suppressPackageStartupMessages(library(spam))
suppressPackageStartupMessages(library(dotCall64))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(DMwR))
suppressPackageStartupMessages(library(uba))
suppressPackageStartupMessages(library(UBL))
suppressPackageStartupMessages(library(MBA))
suppressPackageStartupMessages(library(gstat))
suppressPackageStartupMessages(library(automap))
suppressPackageStartupMessages(library(sp))
suppressPackageStartupMessages(library(randomForest))

library(uba, warn.conflicts = FALSE)  # util used below
library(performanceEstimation)
library(UBL, warn.conflicts = FALSE)
library(ggplot2, warn.conflicts = FALSE)
library(DMwR)
source('DIBSRegress.R')


create_formula <- function(target_variable){
    y <- target_variable
    fmla = as.formula(paste(y, '~ .'))
    return(fmla)
}


WFRandUnder <- function(fmla, train, rel, thrrel, Cperc, repl){
    undersampled = RandUnderRegress(fmla, train, rel, thrrel, Cperc, repl)
    return(undersampled)
}

WFSMOTE <- function(fmla, train, rel, thrrel, Cperc, k, repl, dist, p){
    smoted <- SmoteRegress(fmla, train, rel, thrrel, Cperc, k, repl, dist, p)
    return(smoted)
}

WFGN <- function(fmla, train, rel, thrrel, Cperc, pert, repl){
    gaussnoise <- GaussNoiseRegress(fmla, train, rel, thrrel, Cperc, pert, repl)
    return(gaussnoise)
}


WFDIBS <- function(fmla, dat, method, npts, controlpts, thrrel, Cperc, k, repl, dist, p, pert){
    smogned <- DIBSRegress(fmla, dat, method, npts, controlpts, thrrel, Cperc, k, repl, dist, p, pert)
    return(smogned)
}


get_relevance_params_range <- function(target_variable, rel_method, extr_type='high', coef=1.5, relevance_pts){
    # this function lets the user pass either the method type (extremes or range) to the phi.control function
    # in order to get the attributes of the relevance function phi.
    # If user sets method = range, he/she must set control.pts to be the relevance matrix having three columns

    # setting the target variable and the formula
    y <- target_variable
    phiF.argsR <- phi.control(y,method=rel_method, extr.type = extr_type, coef = coef, control.pts=relevance_pts)
    relevance_values <- phi(y, control.parms = phiF.argsR)
    # ymin, ymax, tloss, epsilon
    lossF.args <- loss.control(y)

    return(list("relevance_params" = phiF.argsR, "loss_params" = lossF.args, "relevance_values"=relevance_values))
}

get_relevance_params_extremes <- function(target_variable, rel_method, extr_type='high', coef=1.5){
    # this function lets the user pass either the method type (extremes or range) to the phi.control function
    # in order to get the attributes of the relevance function phi.
    # If user sets method = extremes, he/she must set extr.type to 'both' (the default), or 'low' or 'high'

    # setting the target variable and the formula
    y <- target_variable
    phiF.argsR <- phi.control(y,method=rel_method, extr.type = extr_type, coef = coef, control.pts=NULL)
    relevance_values <- phi(y, control.parms = phiF.argsR)

    # ymin, ymax, tloss, epsilon
    lossF.args <- loss.control(y)

    return(list("relevance_params" = phiF.argsR, "loss_params" = lossF.args, "relevance_values"=relevance_values))
}


################################################################################################################
# Function to be called when evaluating between actual and predicted in Cross Validation + re-train and testing
###############################################################################################################

eval_stats <- function(trues, preds,
                       thrrel, method,npts,controlpts,
                       ymin,ymax,tloss,epsilon){
  pc <- list()
  pc$method <- method
  pc$npts <- npts
  pc$control.pts <- controlpts
  lossF.args <- list()
  lossF.args$ymin <- ymin
  lossF.args$ymax <- ymax
  lossF.args$tloss <- tloss
  lossF.args$epsilon <- epsilon

  MU <- util(preds, trues, pc, lossF.args, util.control(umetric="MU",p=0.5))
  NMU <- util(preds, trues, pc, lossF.args, util.control(umetric="NMU",p=0.5))
  ubaprec <- util(preds,trues,pc,lossF.args,util.control(umetric="P", event.thr=thrrel, p=0.5))
  ubarec  <- util(preds,trues,pc,lossF.args,util.control(umetric="R", event.thr=thrrel, p=0.5))
  ubaF1   <- util(preds,trues,pc,lossF.args,util.control(umetric="Fm",beta=1, event.thr=thrrel, p=0.5))
  ubaF05   <- util(preds,trues,pc,lossF.args,util.control(umetric="Fm",beta=0.5, event.thr=thrrel, p=0.5))
  ubaF2   <- util(preds,trues,pc,lossF.args,util.control(umetric="Fm",beta=2, event.thr=thrrel, p=0.5))

    return(list(mad = mean(abs(trues-preds)), mse = mean((trues-preds)^2),
                ubaF1 = ubaF1,
                ubaF05 = ubaF05,
                ubaF2 = ubaF2,
                ubaprec = ubaprec,
                ubarec = ubarec,
                MU = MU,
                NMU = NMU))
}

# function for retrieving the relevance values of the target variable in order
# to use it for the function plot_relevance in relevance_helper.py
get_yrel <- function(y, meth, extr_type, control_pts=NULL){
    # check for error in the passed parameters
    # Although I have checked these in the Python function call, I will double check here again :)
    # "frustrated programmer"

    if(meth != 'extremes' & meth != 'range'){
        stop("meth can be either 'range' or 'extremes'")
    }

    else if(meth == 'extermes'){
        if(extr_type != 'high' | extr_type != 'low' | extr_type != 'both'){
            stop("extr_type must be either: 'high', 'low', or 'both'")
        }
    }

    else if(meth == 'range' & is.null(control_pts)){
        stop("with meth = 'range', control_pts must not be NULL")
    }

    else{
        # the plotting part
        if (meth == 'extremes'){
            # without relevance matrix
            phiargs <- phi.control(y,method=meth,extr.type=extr_type)
        }
        else{
            # with relevance matrix
            phiargs <- phi.control(y,method=meth, control.pts=control_pts)
        }

        # the vector containing the relevance values for each value in 'y'
        yphi <- phi(y, control.parms=phiargs)

        return(yphi)
    }
}