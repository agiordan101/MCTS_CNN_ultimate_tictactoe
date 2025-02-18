��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
�
conv_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameconv_layer/kernel
�
%conv_layer/kernel/Read/ReadVariableOpReadVariableOpconv_layer/kernel*'
_output_shapes
:�*
dtype0
w
conv_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv_layer/bias
p
#conv_layer/bias/Read/ReadVariableOpReadVariableOpconv_layer/bias*
_output_shapes	
:�*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
�	�*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:�*
dtype0
x
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense2/kernel
q
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel* 
_output_shapes
:
��*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:�*
dtype0
{
policies/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�Q* 
shared_namepolicies/kernel
t
#policies/kernel/Read/ReadVariableOpReadVariableOppolicies/kernel*
_output_shapes
:	�Q*
dtype0
r
policies/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*
shared_namepolicies/bias
k
!policies/bias/Read/ReadVariableOpReadVariableOppolicies/bias*
_output_shapes
:Q*
dtype0
w
values/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namevalues/kernel
p
!values/kernel/Read/ReadVariableOpReadVariableOpvalues/kernel*
_output_shapes
:	�*
dtype0
n
values/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namevalues/bias
g
values/bias/Read/ReadVariableOpReadVariableOpvalues/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer/kernel/m
�
,Adam/conv_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer/kernel/m*'
_output_shapes
:�*
dtype0
�
Adam/conv_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/conv_layer/bias/m
~
*Adam/conv_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*%
shared_nameAdam/dense1/kernel/m

(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m* 
_output_shapes
:
�	�*
dtype0
}
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dense1/bias/m
v
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dense2/kernel/m

(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dense2/bias/m
v
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/policies/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�Q*'
shared_nameAdam/policies/kernel/m
�
*Adam/policies/kernel/m/Read/ReadVariableOpReadVariableOpAdam/policies/kernel/m*
_output_shapes
:	�Q*
dtype0
�
Adam/policies/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*%
shared_nameAdam/policies/bias/m
y
(Adam/policies/bias/m/Read/ReadVariableOpReadVariableOpAdam/policies/bias/m*
_output_shapes
:Q*
dtype0
�
Adam/conv_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer/kernel/v
�
,Adam/conv_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer/kernel/v*'
_output_shapes
:�*
dtype0
�
Adam/conv_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/conv_layer/bias/v
~
*Adam/conv_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*%
shared_nameAdam/dense1/kernel/v

(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v* 
_output_shapes
:
�	�*
dtype0
}
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dense1/bias/v
v
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dense2/kernel/v

(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dense2/bias/v
v
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/policies/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�Q*'
shared_nameAdam/policies/kernel/v
�
*Adam/policies/kernel/v/Read/ReadVariableOpReadVariableOpAdam/policies/kernel/v*
_output_shapes
:	�Q*
dtype0
�
Adam/policies/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*%
shared_nameAdam/policies/bias/v
y
(Adam/policies/bias/v/Read/ReadVariableOpReadVariableOpAdam/policies/bias/v*
_output_shapes
:Q*
dtype0

NoOpNoOp
�7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�6
value�6B�6 B�6
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	loss


signatures
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
%
#_self_saveable_object_factories
�

kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
�

kernel
bias
#_self_saveable_object_factories
 trainable_variables
!	variables
"regularization_losses
#	keras_api
�

$kernel
%bias
#&_self_saveable_object_factories
'trainable_variables
(	variables
)regularization_losses
*	keras_api
�

+kernel
,bias
#-_self_saveable_object_factories
.trainable_variables
/	variables
0regularization_losses
1	keras_api
�

2kernel
3bias
#4_self_saveable_object_factories
5trainable_variables
6	variables
7regularization_losses
8	keras_api
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratemkmlmmmn$mo%mp+mq,mrvsvtvuvv$vw%vx+vy,vz
 
 
 
F
0
1
2
3
$4
%5
+6
,7
28
39
F
0
1
2
3
$4
%5
+6
,7
28
39
 
�
>non_trainable_variables
?metrics

@layers
Alayer_regularization_losses
trainable_variables
Blayer_metrics
	variables
regularization_losses
 
][
VARIABLE_VALUEconv_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
Cnon_trainable_variables
Dmetrics

Elayers
Flayer_regularization_losses
trainable_variables
Glayer_metrics
	variables
regularization_losses
 
 
 
 
�
Hnon_trainable_variables
Imetrics

Jlayers
Klayer_regularization_losses
trainable_variables
Llayer_metrics
	variables
regularization_losses
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
Mnon_trainable_variables
Nmetrics

Olayers
Player_regularization_losses
 trainable_variables
Qlayer_metrics
!	variables
"regularization_losses
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
 
�
Rnon_trainable_variables
Smetrics

Tlayers
Ulayer_regularization_losses
'trainable_variables
Vlayer_metrics
(	variables
)regularization_losses
[Y
VARIABLE_VALUEpolicies/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEpolicies/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
 
�
Wnon_trainable_variables
Xmetrics

Ylayers
Zlayer_regularization_losses
.trainable_variables
[layer_metrics
/	variables
0regularization_losses
YW
VARIABLE_VALUEvalues/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEvalues/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
 
�
\non_trainable_variables
]metrics

^layers
_layer_regularization_losses
5trainable_variables
`layer_metrics
6	variables
7regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ctotal
	dcount
e	variables
f	keras_api
4
	gtotal
	hcount
i	variables
j	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

e	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

i	variables
�~
VARIABLE_VALUEAdam/conv_layer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_layer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/policies/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/policies/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv_layer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_layer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/policies/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/policies/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������		*
dtype0*$
shape:���������		
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_layer/kernelconv_layer/biasdense1/kerneldense1/biasdense2/kerneldense2/biasvalues/kernelvalues/biaspolicies/kernelpolicies/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_4508555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv_layer/kernel/Read/ReadVariableOp#conv_layer/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp#policies/kernel/Read/ReadVariableOp!policies/bias/Read/ReadVariableOp!values/kernel/Read/ReadVariableOpvalues/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv_layer/kernel/m/Read/ReadVariableOp*Adam/conv_layer/bias/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp*Adam/policies/kernel/m/Read/ReadVariableOp(Adam/policies/bias/m/Read/ReadVariableOp,Adam/conv_layer/kernel/v/Read/ReadVariableOp*Adam/conv_layer/bias/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp*Adam/policies/kernel/v/Read/ReadVariableOp(Adam/policies/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_4508933
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_layer/kernelconv_layer/biasdense1/kerneldense1/biasdense2/kerneldense2/biaspolicies/kernelpolicies/biasvalues/kernelvalues/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv_layer/kernel/mAdam/conv_layer/bias/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/policies/kernel/mAdam/policies/bias/mAdam/conv_layer/kernel/vAdam/conv_layer/bias/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense2/kernel/vAdam/dense2/bias/vAdam/policies/kernel/vAdam/policies/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_4509048��
�!
�
B__inference_model_layer_call_and_return_conditional_losses_4508435

inputs
conv_layer_4508407
conv_layer_4508409
dense1_4508413
dense1_4508415
dense2_4508418
dense2_4508420
values_4508423
values_4508425
policies_4508428
policies_4508430
identity

identity_1��"conv_layer/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall� policies/StatefulPartitionedCall�values/StatefulPartitionedCall�
"conv_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_4508407conv_layer_4508409*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_layer_layer_call_and_return_conditional_losses_45082302$
"conv_layer/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall+conv_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_45082522
flatten/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4508413dense1_4508415*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_45082712 
dense1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4508418dense2_4508420*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_45082982 
dense2/StatefulPartitionedCall�
values/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0values_4508423values_4508425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_values_layer_call_and_return_conditional_losses_45083252 
values/StatefulPartitionedCall�
 policies/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0policies_4508428policies_4508430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_policies_layer_call_and_return_conditional_losses_45083522"
 policies/StatefulPartitionedCall�
IdentityIdentity)policies/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity'values/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2H
"conv_layer/StatefulPartitionedCall"conv_layer/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 policies/StatefulPartitionedCall policies/StatefulPartitionedCall2@
values/StatefulPartitionedCallvalues/StatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�	
�
C__inference_dense2_layer_call_and_return_conditional_losses_4508755

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�<
�
"__inference__wrapped_model_4508215
input_13
/model_conv_layer_conv2d_readvariableop_resource4
0model_conv_layer_biasadd_readvariableop_resource/
+model_dense1_matmul_readvariableop_resource0
,model_dense1_biasadd_readvariableop_resource/
+model_dense2_matmul_readvariableop_resource0
,model_dense2_biasadd_readvariableop_resource/
+model_values_matmul_readvariableop_resource0
,model_values_biasadd_readvariableop_resource1
-model_policies_matmul_readvariableop_resource2
.model_policies_biasadd_readvariableop_resource
identity

identity_1��'model/conv_layer/BiasAdd/ReadVariableOp�&model/conv_layer/Conv2D/ReadVariableOp�#model/dense1/BiasAdd/ReadVariableOp�"model/dense1/MatMul/ReadVariableOp�#model/dense2/BiasAdd/ReadVariableOp�"model/dense2/MatMul/ReadVariableOp�%model/policies/BiasAdd/ReadVariableOp�$model/policies/MatMul/ReadVariableOp�#model/values/BiasAdd/ReadVariableOp�"model/values/MatMul/ReadVariableOp�
&model/conv_layer/Conv2D/ReadVariableOpReadVariableOp/model_conv_layer_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02(
&model/conv_layer/Conv2D/ReadVariableOp�
model/conv_layer/Conv2DConv2Dinput_1.model/conv_layer/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
model/conv_layer/Conv2D�
'model/conv_layer/BiasAdd/ReadVariableOpReadVariableOp0model_conv_layer_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model/conv_layer/BiasAdd/ReadVariableOp�
model/conv_layer/BiasAddBiasAdd model/conv_layer/Conv2D:output:0/model/conv_layer/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model/conv_layer/BiasAdd�
model/conv_layer/ReluRelu!model/conv_layer/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model/conv_layer/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model/flatten/Const�
model/flatten/ReshapeReshape#model/conv_layer/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������	2
model/flatten/Reshape�
"model/dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02$
"model/dense1/MatMul/ReadVariableOp�
model/dense1/MatMulMatMulmodel/flatten/Reshape:output:0*model/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense1/MatMul�
#model/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#model/dense1/BiasAdd/ReadVariableOp�
model/dense1/BiasAddBiasAddmodel/dense1/MatMul:product:0+model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense1/BiasAdd�
model/dense1/ReluRelumodel/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/dense1/Relu�
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02$
"model/dense2/MatMul/ReadVariableOp�
model/dense2/MatMulMatMulmodel/dense1/Relu:activations:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense2/MatMul�
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#model/dense2/BiasAdd/ReadVariableOp�
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense2/BiasAdd�
model/dense2/ReluRelumodel/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/dense2/Relu�
"model/values/MatMul/ReadVariableOpReadVariableOp+model_values_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"model/values/MatMul/ReadVariableOp�
model/values/MatMulMatMulmodel/dense2/Relu:activations:0*model/values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/values/MatMul�
#model/values/BiasAdd/ReadVariableOpReadVariableOp,model_values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/values/BiasAdd/ReadVariableOp�
model/values/BiasAddBiasAddmodel/values/MatMul:product:0+model/values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/values/BiasAdd
model/values/TanhTanhmodel/values/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/values/Tanh�
$model/policies/MatMul/ReadVariableOpReadVariableOp-model_policies_matmul_readvariableop_resource*
_output_shapes
:	�Q*
dtype02&
$model/policies/MatMul/ReadVariableOp�
model/policies/MatMulMatMulmodel/dense2/Relu:activations:0,model/policies/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
model/policies/MatMul�
%model/policies/BiasAdd/ReadVariableOpReadVariableOp.model_policies_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02'
%model/policies/BiasAdd/ReadVariableOp�
model/policies/BiasAddBiasAddmodel/policies/MatMul:product:0-model/policies/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
model/policies/BiasAdd�
model/policies/SoftmaxSoftmaxmodel/policies/BiasAdd:output:0*
T0*'
_output_shapes
:���������Q2
model/policies/Softmax�
IdentityIdentity model/policies/Softmax:softmax:0(^model/conv_layer/BiasAdd/ReadVariableOp'^model/conv_layer/Conv2D/ReadVariableOp$^model/dense1/BiasAdd/ReadVariableOp#^model/dense1/MatMul/ReadVariableOp$^model/dense2/BiasAdd/ReadVariableOp#^model/dense2/MatMul/ReadVariableOp&^model/policies/BiasAdd/ReadVariableOp%^model/policies/MatMul/ReadVariableOp$^model/values/BiasAdd/ReadVariableOp#^model/values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identitymodel/values/Tanh:y:0(^model/conv_layer/BiasAdd/ReadVariableOp'^model/conv_layer/Conv2D/ReadVariableOp$^model/dense1/BiasAdd/ReadVariableOp#^model/dense1/MatMul/ReadVariableOp$^model/dense2/BiasAdd/ReadVariableOp#^model/dense2/MatMul/ReadVariableOp&^model/policies/BiasAdd/ReadVariableOp%^model/policies/MatMul/ReadVariableOp$^model/values/BiasAdd/ReadVariableOp#^model/values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2R
'model/conv_layer/BiasAdd/ReadVariableOp'model/conv_layer/BiasAdd/ReadVariableOp2P
&model/conv_layer/Conv2D/ReadVariableOp&model/conv_layer/Conv2D/ReadVariableOp2J
#model/dense1/BiasAdd/ReadVariableOp#model/dense1/BiasAdd/ReadVariableOp2H
"model/dense1/MatMul/ReadVariableOp"model/dense1/MatMul/ReadVariableOp2J
#model/dense2/BiasAdd/ReadVariableOp#model/dense2/BiasAdd/ReadVariableOp2H
"model/dense2/MatMul/ReadVariableOp"model/dense2/MatMul/ReadVariableOp2N
%model/policies/BiasAdd/ReadVariableOp%model/policies/BiasAdd/ReadVariableOp2L
$model/policies/MatMul/ReadVariableOp$model/policies/MatMul/ReadVariableOp2J
#model/values/BiasAdd/ReadVariableOp#model/values/BiasAdd/ReadVariableOp2H
"model/values/MatMul/ReadVariableOp"model/values/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�	
�
E__inference_policies_layer_call_and_return_conditional_losses_4508775

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�Q*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������Q2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_policies_layer_call_fn_4508784

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_policies_layer_call_and_return_conditional_losses_45083522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�
B__inference_model_layer_call_and_return_conditional_losses_4508639

inputs-
)conv_layer_conv2d_readvariableop_resource.
*conv_layer_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%values_matmul_readvariableop_resource*
&values_biasadd_readvariableop_resource+
'policies_matmul_readvariableop_resource,
(policies_biasadd_readvariableop_resource
identity

identity_1��!conv_layer/BiasAdd/ReadVariableOp� conv_layer/Conv2D/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�dense2/BiasAdd/ReadVariableOp�dense2/MatMul/ReadVariableOp�policies/BiasAdd/ReadVariableOp�policies/MatMul/ReadVariableOp�values/BiasAdd/ReadVariableOp�values/MatMul/ReadVariableOp�
 conv_layer/Conv2D/ReadVariableOpReadVariableOp)conv_layer_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02"
 conv_layer/Conv2D/ReadVariableOp�
conv_layer/Conv2DConv2Dinputs(conv_layer/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv_layer/Conv2D�
!conv_layer/BiasAdd/ReadVariableOpReadVariableOp*conv_layer_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv_layer/BiasAdd/ReadVariableOp�
conv_layer/BiasAddBiasAddconv_layer/Conv2D:output:0)conv_layer/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv_layer/BiasAdd�
conv_layer/ReluReluconv_layer/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv_layer/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten/Const�
flatten/ReshapeReshapeconv_layer/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	2
flatten/Reshape�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
dense1/MatMul/ReadVariableOp�
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense1/MatMul�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense1/Relu�
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense2/MatMul/ReadVariableOp�
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense2/MatMul�
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense2/BiasAdd/ReadVariableOp�
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense2/Relu�
values/MatMul/ReadVariableOpReadVariableOp%values_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
values/MatMul/ReadVariableOp�
values/MatMulMatMuldense2/Relu:activations:0$values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
values/MatMul�
values/BiasAdd/ReadVariableOpReadVariableOp&values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
values/BiasAdd/ReadVariableOp�
values/BiasAddBiasAddvalues/MatMul:product:0%values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
values/BiasAddm
values/TanhTanhvalues/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
values/Tanh�
policies/MatMul/ReadVariableOpReadVariableOp'policies_matmul_readvariableop_resource*
_output_shapes
:	�Q*
dtype02 
policies/MatMul/ReadVariableOp�
policies/MatMulMatMuldense2/Relu:activations:0&policies/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
policies/MatMul�
policies/BiasAdd/ReadVariableOpReadVariableOp(policies_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02!
policies/BiasAdd/ReadVariableOp�
policies/BiasAddBiasAddpolicies/MatMul:product:0'policies/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
policies/BiasAdd|
policies/SoftmaxSoftmaxpolicies/BiasAdd:output:0*
T0*'
_output_shapes
:���������Q2
policies/Softmax�
IdentityIdentitypolicies/Softmax:softmax:0"^conv_layer/BiasAdd/ReadVariableOp!^conv_layer/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp ^policies/BiasAdd/ReadVariableOp^policies/MatMul/ReadVariableOp^values/BiasAdd/ReadVariableOp^values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identityvalues/Tanh:y:0"^conv_layer/BiasAdd/ReadVariableOp!^conv_layer/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp ^policies/BiasAdd/ReadVariableOp^policies/MatMul/ReadVariableOp^values/BiasAdd/ReadVariableOp^values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2F
!conv_layer/BiasAdd/ReadVariableOp!conv_layer/BiasAdd/ReadVariableOp2D
 conv_layer/Conv2D/ReadVariableOp conv_layer/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2B
policies/BiasAdd/ReadVariableOppolicies/BiasAdd/ReadVariableOp2@
policies/MatMul/ReadVariableOppolicies/MatMul/ReadVariableOp2>
values/BiasAdd/ReadVariableOpvalues/BiasAdd/ReadVariableOp2<
values/MatMul/ReadVariableOpvalues/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�	
�
C__inference_values_layer_call_and_return_conditional_losses_4508795

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_values_layer_call_and_return_conditional_losses_4508325

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_4508370
input_1
conv_layer_4508241
conv_layer_4508243
dense1_4508282
dense1_4508284
dense2_4508309
dense2_4508311
values_4508336
values_4508338
policies_4508363
policies_4508365
identity

identity_1��"conv_layer/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall� policies/StatefulPartitionedCall�values/StatefulPartitionedCall�
"conv_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_layer_4508241conv_layer_4508243*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_layer_layer_call_and_return_conditional_losses_45082302$
"conv_layer/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall+conv_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_45082522
flatten/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4508282dense1_4508284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_45082712 
dense1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4508309dense2_4508311*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_45082982 
dense2/StatefulPartitionedCall�
values/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0values_4508336values_4508338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_values_layer_call_and_return_conditional_losses_45083252 
values/StatefulPartitionedCall�
 policies/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0policies_4508363policies_4508365*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_policies_layer_call_and_return_conditional_losses_45083522"
 policies/StatefulPartitionedCall�
IdentityIdentity)policies/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity'values/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2H
"conv_layer/StatefulPartitionedCall"conv_layer/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 policies/StatefulPartitionedCall policies/StatefulPartitionedCall2@
values/StatefulPartitionedCallvalues/StatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�

�
G__inference_conv_layer_layer_call_and_return_conditional_losses_4508704

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�5
�
B__inference_model_layer_call_and_return_conditional_losses_4508597

inputs-
)conv_layer_conv2d_readvariableop_resource.
*conv_layer_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%values_matmul_readvariableop_resource*
&values_biasadd_readvariableop_resource+
'policies_matmul_readvariableop_resource,
(policies_biasadd_readvariableop_resource
identity

identity_1��!conv_layer/BiasAdd/ReadVariableOp� conv_layer/Conv2D/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�dense2/BiasAdd/ReadVariableOp�dense2/MatMul/ReadVariableOp�policies/BiasAdd/ReadVariableOp�policies/MatMul/ReadVariableOp�values/BiasAdd/ReadVariableOp�values/MatMul/ReadVariableOp�
 conv_layer/Conv2D/ReadVariableOpReadVariableOp)conv_layer_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02"
 conv_layer/Conv2D/ReadVariableOp�
conv_layer/Conv2DConv2Dinputs(conv_layer/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv_layer/Conv2D�
!conv_layer/BiasAdd/ReadVariableOpReadVariableOp*conv_layer_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv_layer/BiasAdd/ReadVariableOp�
conv_layer/BiasAddBiasAddconv_layer/Conv2D:output:0)conv_layer/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv_layer/BiasAdd�
conv_layer/ReluReluconv_layer/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv_layer/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten/Const�
flatten/ReshapeReshapeconv_layer/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	2
flatten/Reshape�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
dense1/MatMul/ReadVariableOp�
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense1/MatMul�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense1/Relu�
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense2/MatMul/ReadVariableOp�
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense2/MatMul�
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense2/BiasAdd/ReadVariableOp�
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense2/Relu�
values/MatMul/ReadVariableOpReadVariableOp%values_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
values/MatMul/ReadVariableOp�
values/MatMulMatMuldense2/Relu:activations:0$values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
values/MatMul�
values/BiasAdd/ReadVariableOpReadVariableOp&values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
values/BiasAdd/ReadVariableOp�
values/BiasAddBiasAddvalues/MatMul:product:0%values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
values/BiasAddm
values/TanhTanhvalues/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
values/Tanh�
policies/MatMul/ReadVariableOpReadVariableOp'policies_matmul_readvariableop_resource*
_output_shapes
:	�Q*
dtype02 
policies/MatMul/ReadVariableOp�
policies/MatMulMatMuldense2/Relu:activations:0&policies/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
policies/MatMul�
policies/BiasAdd/ReadVariableOpReadVariableOp(policies_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02!
policies/BiasAdd/ReadVariableOp�
policies/BiasAddBiasAddpolicies/MatMul:product:0'policies/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
policies/BiasAdd|
policies/SoftmaxSoftmaxpolicies/BiasAdd:output:0*
T0*'
_output_shapes
:���������Q2
policies/Softmax�
IdentityIdentitypolicies/Softmax:softmax:0"^conv_layer/BiasAdd/ReadVariableOp!^conv_layer/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp ^policies/BiasAdd/ReadVariableOp^policies/MatMul/ReadVariableOp^values/BiasAdd/ReadVariableOp^values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identityvalues/Tanh:y:0"^conv_layer/BiasAdd/ReadVariableOp!^conv_layer/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp ^policies/BiasAdd/ReadVariableOp^policies/MatMul/ReadVariableOp^values/BiasAdd/ReadVariableOp^values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2F
!conv_layer/BiasAdd/ReadVariableOp!conv_layer/BiasAdd/ReadVariableOp2D
 conv_layer/Conv2D/ReadVariableOp conv_layer/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2B
policies/BiasAdd/ReadVariableOppolicies/BiasAdd/ReadVariableOp2@
policies/MatMul/ReadVariableOppolicies/MatMul/ReadVariableOp2>
values/BiasAdd/ReadVariableOpvalues/BiasAdd/ReadVariableOp2<
values/MatMul/ReadVariableOpvalues/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_4508401
input_1
conv_layer_4508373
conv_layer_4508375
dense1_4508379
dense1_4508381
dense2_4508384
dense2_4508386
values_4508389
values_4508391
policies_4508394
policies_4508396
identity

identity_1��"conv_layer/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall� policies/StatefulPartitionedCall�values/StatefulPartitionedCall�
"conv_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_layer_4508373conv_layer_4508375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_layer_layer_call_and_return_conditional_losses_45082302$
"conv_layer/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall+conv_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_45082522
flatten/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4508379dense1_4508381*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_45082712 
dense1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4508384dense2_4508386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_45082982 
dense2/StatefulPartitionedCall�
values/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0values_4508389values_4508391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_values_layer_call_and_return_conditional_losses_45083252 
values/StatefulPartitionedCall�
 policies/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0policies_4508394policies_4508396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_policies_layer_call_and_return_conditional_losses_45083522"
 policies/StatefulPartitionedCall�
IdentityIdentity)policies/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity'values/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2H
"conv_layer/StatefulPartitionedCall"conv_layer/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 policies/StatefulPartitionedCall policies/StatefulPartitionedCall2@
values/StatefulPartitionedCallvalues/StatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�
}
(__inference_dense1_layer_call_fn_4508744

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_45082712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�

�
'__inference_model_layer_call_fn_4508460
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_45084352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�

�
%__inference_signature_wrapper_4508555
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_45082152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�K
�
 __inference__traced_save_4508933
file_prefix0
,savev2_conv_layer_kernel_read_readvariableop.
*savev2_conv_layer_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop.
*savev2_policies_kernel_read_readvariableop,
(savev2_policies_bias_read_readvariableop,
(savev2_values_kernel_read_readvariableop*
&savev2_values_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv_layer_kernel_m_read_readvariableop5
1savev2_adam_conv_layer_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop5
1savev2_adam_policies_kernel_m_read_readvariableop3
/savev2_adam_policies_bias_m_read_readvariableop7
3savev2_adam_conv_layer_kernel_v_read_readvariableop5
1savev2_adam_conv_layer_bias_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop5
1savev2_adam_policies_kernel_v_read_readvariableop3
/savev2_adam_policies_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv_layer_kernel_read_readvariableop*savev2_conv_layer_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop*savev2_policies_kernel_read_readvariableop(savev2_policies_bias_read_readvariableop(savev2_values_kernel_read_readvariableop&savev2_values_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv_layer_kernel_m_read_readvariableop1savev2_adam_conv_layer_bias_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop1savev2_adam_policies_kernel_m_read_readvariableop/savev2_adam_policies_bias_m_read_readvariableop3savev2_adam_conv_layer_kernel_v_read_readvariableop1savev2_adam_conv_layer_bias_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop1savev2_adam_policies_kernel_v_read_readvariableop/savev2_adam_policies_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:
�	�:�:
��:�:	�Q:Q:	�:: : : : : : : : : :�:�:
�	�:�:
��:�:	�Q:Q:�:�:
�	�:�:
��:�:	�Q:Q: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�Q: 

_output_shapes
:Q:%	!

_output_shapes
:	�: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�Q: 

_output_shapes
:Q:-)
'
_output_shapes
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:& "
 
_output_shapes
:
��:!!

_output_shapes	
:�:%"!

_output_shapes
:	�Q: #

_output_shapes
:Q:$

_output_shapes
: 
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_4508252

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_dense2_layer_call_fn_4508764

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_45082982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_model_layer_call_fn_4508693

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_45084932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�

�
G__inference_conv_layer_layer_call_and_return_conditional_losses_4508230

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�

�
'__inference_model_layer_call_fn_4508666

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_45084352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�	
�
C__inference_dense1_layer_call_and_return_conditional_losses_4508271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_4508493

inputs
conv_layer_4508465
conv_layer_4508467
dense1_4508471
dense1_4508473
dense2_4508476
dense2_4508478
values_4508481
values_4508483
policies_4508486
policies_4508488
identity

identity_1��"conv_layer/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall� policies/StatefulPartitionedCall�values/StatefulPartitionedCall�
"conv_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_4508465conv_layer_4508467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_layer_layer_call_and_return_conditional_losses_45082302$
"conv_layer/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall+conv_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_45082522
flatten/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4508471dense1_4508473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_45082712 
dense1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4508476dense2_4508478*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_45082982 
dense2/StatefulPartitionedCall�
values/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0values_4508481values_4508483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_values_layer_call_and_return_conditional_losses_45083252 
values/StatefulPartitionedCall�
 policies/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0policies_4508486policies_4508488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_policies_layer_call_and_return_conditional_losses_45083522"
 policies/StatefulPartitionedCall�
IdentityIdentity)policies/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity'values/StatefulPartitionedCall:output:0#^conv_layer/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^policies/StatefulPartitionedCall^values/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::2H
"conv_layer/StatefulPartitionedCall"conv_layer/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 policies/StatefulPartitionedCall policies/StatefulPartitionedCall2@
values/StatefulPartitionedCallvalues/StatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_4509048
file_prefix&
"assignvariableop_conv_layer_kernel&
"assignvariableop_1_conv_layer_bias$
 assignvariableop_2_dense1_kernel"
assignvariableop_3_dense1_bias$
 assignvariableop_4_dense2_kernel"
assignvariableop_5_dense2_bias&
"assignvariableop_6_policies_kernel$
 assignvariableop_7_policies_bias$
 assignvariableop_8_values_kernel"
assignvariableop_9_values_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_10
,assignvariableop_19_adam_conv_layer_kernel_m.
*assignvariableop_20_adam_conv_layer_bias_m,
(assignvariableop_21_adam_dense1_kernel_m*
&assignvariableop_22_adam_dense1_bias_m,
(assignvariableop_23_adam_dense2_kernel_m*
&assignvariableop_24_adam_dense2_bias_m.
*assignvariableop_25_adam_policies_kernel_m,
(assignvariableop_26_adam_policies_bias_m0
,assignvariableop_27_adam_conv_layer_kernel_v.
*assignvariableop_28_adam_conv_layer_bias_v,
(assignvariableop_29_adam_dense1_kernel_v*
&assignvariableop_30_adam_dense1_bias_v,
(assignvariableop_31_adam_dense2_kernel_v*
&assignvariableop_32_adam_dense2_bias_v.
*assignvariableop_33_adam_policies_kernel_v,
(assignvariableop_34_adam_policies_bias_v
identity_36��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp"assignvariableop_conv_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_policies_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_policies_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_values_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_values_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv_layer_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv_layer_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_dense1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_dense2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_policies_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_policies_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_conv_layer_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv_layer_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_dense1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_dense2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_policies_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_policies_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35�
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
C__inference_dense1_layer_call_and_return_conditional_losses_4508735

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
,__inference_conv_layer_layer_call_fn_4508713

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_layer_layer_call_and_return_conditional_losses_45082302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_4508724

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_45082522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_4508719

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_dense2_layer_call_and_return_conditional_losses_4508298

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_model_layer_call_fn_4508518
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������Q:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_45084932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������Q2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:���������		::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1
�	
�
E__inference_policies_layer_call_and_return_conditional_losses_4508352

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�Q*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������Q2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_values_layer_call_fn_4508804

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_values_layer_call_and_return_conditional_losses_45083252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������		<
policies0
StatefulPartitionedCall:0���������Q:
values0
StatefulPartitionedCall:1���������tensorflow/serving/predict:��
�=
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	loss


signatures
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|_default_save_signature
}__call__"�:
_tf_keras_network�9{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_layer", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "policies", "trainable": true, "dtype": "float32", "units": 81, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "policies", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "values", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "values", "inbound_nodes": [[["dense2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["policies", 0, 0], ["values", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9, 9, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_layer", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "policies", "trainable": true, "dtype": "float32", "units": 81, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "policies", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "values", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "values", "inbound_nodes": [[["dense2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["policies", 0, 0], ["values", 0, 0]]}}, "training_config": {"loss": ["mean_squared_error"], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
#_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�


kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
*~&call_and_return_all_conditional_losses
__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_layer", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 3]}}
�
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
#_self_saveable_object_factories
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
�

$kernel
%bias
#&_self_saveable_object_factories
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
�

+kernel
,bias
#-_self_saveable_object_factories
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "policies", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "policies", "trainable": true, "dtype": "float32", "units": 81, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

2kernel
3bias
#4_self_saveable_object_factories
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "values", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "values", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratemkmlmmmn$mo%mp+mq,mrvsvtvuvv$vw%vx+vy,vz"
	optimizer
 "
trackable_list_wrapper
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
f
0
1
2
3
$4
%5
+6
,7
28
39"
trackable_list_wrapper
f
0
1
2
3
$4
%5
+6
,7
28
39"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables
?metrics

@layers
Alayer_regularization_losses
trainable_variables
Blayer_metrics
	variables
regularization_losses
}__call__
|_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
,:*�2conv_layer/kernel
:�2conv_layer/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables
Dmetrics

Elayers
Flayer_regularization_losses
trainable_variables
Glayer_metrics
	variables
regularization_losses
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hnon_trainable_variables
Imetrics

Jlayers
Klayer_regularization_losses
trainable_variables
Llayer_metrics
	variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
�	�2dense1/kernel
:�2dense1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables
Nmetrics

Olayers
Player_regularization_losses
 trainable_variables
Qlayer_metrics
!	variables
"regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dense2/kernel
:�2dense2/bias
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rnon_trainable_variables
Smetrics

Tlayers
Ulayer_regularization_losses
'trainable_variables
Vlayer_metrics
(	variables
)regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�Q2policies/kernel
:Q2policies/bias
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables
Xmetrics

Ylayers
Zlayer_regularization_losses
.trainable_variables
[layer_metrics
/	variables
0regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2values/kernel
:2values/bias
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables
]metrics

^layers
_layer_regularization_losses
5trainable_variables
`layer_metrics
6	variables
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	ctotal
	dcount
e	variables
f	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	gtotal
	hcount
i	variables
j	keras_api"�
_tf_keras_metric|{"class_name": "Mean", "name": "policies_loss", "dtype": "float32", "config": {"name": "policies_loss", "dtype": "float32"}}
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
.
g0
h1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
1:/�2Adam/conv_layer/kernel/m
#:!�2Adam/conv_layer/bias/m
&:$
�	�2Adam/dense1/kernel/m
:�2Adam/dense1/bias/m
&:$
��2Adam/dense2/kernel/m
:�2Adam/dense2/bias/m
':%	�Q2Adam/policies/kernel/m
 :Q2Adam/policies/bias/m
1:/�2Adam/conv_layer/kernel/v
#:!�2Adam/conv_layer/bias/v
&:$
�	�2Adam/dense1/kernel/v
:�2Adam/dense1/bias/v
&:$
��2Adam/dense2/kernel/v
:�2Adam/dense2/bias/v
':%	�Q2Adam/policies/kernel/v
 :Q2Adam/policies/bias/v
�2�
B__inference_model_layer_call_and_return_conditional_losses_4508639
B__inference_model_layer_call_and_return_conditional_losses_4508597
B__inference_model_layer_call_and_return_conditional_losses_4508370
B__inference_model_layer_call_and_return_conditional_losses_4508401�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_4508215�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������		
�2�
'__inference_model_layer_call_fn_4508518
'__inference_model_layer_call_fn_4508666
'__inference_model_layer_call_fn_4508693
'__inference_model_layer_call_fn_4508460�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_conv_layer_layer_call_and_return_conditional_losses_4508704�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_conv_layer_layer_call_fn_4508713�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_layer_call_and_return_conditional_losses_4508719�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_layer_call_fn_4508724�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense1_layer_call_and_return_conditional_losses_4508735�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense1_layer_call_fn_4508744�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense2_layer_call_and_return_conditional_losses_4508755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense2_layer_call_fn_4508764�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_policies_layer_call_and_return_conditional_losses_4508775�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_policies_layer_call_fn_4508784�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_values_layer_call_and_return_conditional_losses_4508795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_values_layer_call_fn_4508804�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_4508555input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_4508215�
$%23+,8�5
.�+
)�&
input_1���������		
� "_�\
.
policies"�
policies���������Q
*
values �
values����������
G__inference_conv_layer_layer_call_and_return_conditional_losses_4508704m7�4
-�*
(�%
inputs���������		
� ".�+
$�!
0����������
� �
,__inference_conv_layer_layer_call_fn_4508713`7�4
-�*
(�%
inputs���������		
� "!������������
C__inference_dense1_layer_call_and_return_conditional_losses_4508735^0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� }
(__inference_dense1_layer_call_fn_4508744Q0�-
&�#
!�
inputs����������	
� "������������
C__inference_dense2_layer_call_and_return_conditional_losses_4508755^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense2_layer_call_fn_4508764Q$%0�-
&�#
!�
inputs����������
� "������������
D__inference_flatten_layer_call_and_return_conditional_losses_4508719b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������	
� �
)__inference_flatten_layer_call_fn_4508724U8�5
.�+
)�&
inputs����������
� "�����������	�
B__inference_model_layer_call_and_return_conditional_losses_4508370�
$%23+,@�=
6�3
)�&
input_1���������		
p

 
� "K�H
A�>
�
0/0���������Q
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4508401�
$%23+,@�=
6�3
)�&
input_1���������		
p 

 
� "K�H
A�>
�
0/0���������Q
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4508597�
$%23+,?�<
5�2
(�%
inputs���������		
p

 
� "K�H
A�>
�
0/0���������Q
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4508639�
$%23+,?�<
5�2
(�%
inputs���������		
p 

 
� "K�H
A�>
�
0/0���������Q
�
0/1���������
� �
'__inference_model_layer_call_fn_4508460�
$%23+,@�=
6�3
)�&
input_1���������		
p

 
� "=�:
�
0���������Q
�
1����������
'__inference_model_layer_call_fn_4508518�
$%23+,@�=
6�3
)�&
input_1���������		
p 

 
� "=�:
�
0���������Q
�
1����������
'__inference_model_layer_call_fn_4508666�
$%23+,?�<
5�2
(�%
inputs���������		
p

 
� "=�:
�
0���������Q
�
1����������
'__inference_model_layer_call_fn_4508693�
$%23+,?�<
5�2
(�%
inputs���������		
p 

 
� "=�:
�
0���������Q
�
1����������
E__inference_policies_layer_call_and_return_conditional_losses_4508775]+,0�-
&�#
!�
inputs����������
� "%�"
�
0���������Q
� ~
*__inference_policies_layer_call_fn_4508784P+,0�-
&�#
!�
inputs����������
� "����������Q�
%__inference_signature_wrapper_4508555�
$%23+,C�@
� 
9�6
4
input_1)�&
input_1���������		"_�\
.
policies"�
policies���������Q
*
values �
values����������
C__inference_values_layer_call_and_return_conditional_losses_4508795]230�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_values_layer_call_fn_4508804P230�-
&�#
!�
inputs����������
� "����������