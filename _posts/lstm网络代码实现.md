#LSTM网络代码
##keras版本
keras版本就比较简单，基本上就是add和确定layer units的事情

    model = Sequential()
    model.add(LSTM(units=layer_num1, input_shape=(time_step, data_dim)))
    model.add(LSTM(units=layer_num2))
    model.add(Dense(units=num_class))
	model.compile(loss='mean_squared_error', optimizers.Adam(0.01))  # 可选择loss function和学习率和优化方式


##tensorflow版本
	cell = tf.nn.rnn_cell.LSTMCell(num_units)
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_data, initial_state=init_state, dtype=tf.float32)
	# 如果是多对一结构，直接筛选最后一个time_step的结果,因为output_rnn是[batch_size, time_step, num_units], final_states是内部cell的结果，是二维的
	# output = output_rnn[:, -1]    # 多对一的结果筛选
	
	
