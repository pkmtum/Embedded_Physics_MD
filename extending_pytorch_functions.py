import torch
import numpy as np

  #def __init__(self, reduce = False):
  #  super(U, self).__init__()
  #  self.reduce = reduce
  #  print self.reduce
  
class MDSim:
  def __init__(self, strinit):
    self.strinit = strinit
  
class U(torch.autograd.Function):
  
  # This method should provide the potential energy or -log(p(x)). This needs to be decided.
  @staticmethod
  def forward(ctx, input, mdsimulator, reduce = False):
    
    
    output = input.data*input.data
    print output
    #print 'Reduce {}'.format(reduce)
    if reduce or input.dim()==1:
      output = output.sum()
    else:
      output = output.sum(dim=1, keepdim=True)
    ctx.save_for_backward(input)
    ctx.reduce = reduce
    ctx.bDebug = False
    
    #output = torch.tensor(43.)
    
    print mdsimulator.strinit
    
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    #grad = input.mul(2.)
    dtypeinput = input.dtype
    if input.dim() == 1:
      grad = torch.from_numpy(np.array([2., 4., 6.])).type(dtypeinput)
    else:
      grad = torch.from_numpy(np.array([[2., 4., 6.],[4.,6.,8.]])).type(dtypeinput)
    
    gradorig = input.mul(2.)
    grad_input = grad_output*grad
    if ctx.bDebug:
      print 'Grad_output'
      print grad_output
      print 'Grad_input'
      print grad_input
    return grad_input, None, None

#x = torch.autograd.Variable(torch.tensor([1,2,3], dtype=torch.float32, requires_grad = True))
x = torch.tensor([[1,2,3], [2,3,4]], dtype=torch.float32, requires_grad = True)
#x = torch.tensor([1,2,3], dtype=torch.float32, requires_grad = True)
mdsim = MDSim('inits')
#f = uinstance(x, uinstance, True)
f = U.apply(x, mdsim, True)
y = f
z = y.pow(2.)

z.backward(retain_graph=True)
torch.autograd.grad(z,y, retain_graph=True)
torch.autograd.grad(y,x, retain_graph=True)
torch.autograd.grad(z,x, retain_graph=True)
print x.grad

quit() 
