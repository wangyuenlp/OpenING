# OpenING: Open INstruction Generation 



CCL2023 Demo 



1. model training

   ```
   torchrun --nproc_per_node=8 \
   
   --master_port=1234 baichuan_instruction_generation.py \
   
   --model_name_or_path baichuan-inc/Baichuan-7B \
   
   --data_path $DATA_PATH \
   
   --output_dir $OUTPUT_PATH \
   
   --num_train_epochs 3 \
   
   --per_device_train_batch_size 4 \
   
   --per_device_eval_batch_size 4 \
   
   --gradient_accumulation_steps 4 \
   
   --evaluation_strategy "no" \
   
   --save_strategy "steps" \
   
   --save_steps 2000 \
   
   --save_total_limit 1 \
   
   --learning_rate 2e-5 \
   
   --weight_decay 0. \
   
   --warmup_ratio 0.03 \
   
   --lr_scheduler_type "cosine" \
   
   --logging_steps 1 \
   
   --fp16 True \
   
   --report_to 'none'
   ```

   

2. model inference

   ```
   for template_id in {0,1,2}
   
   do
   
   python baichuan_generation.py \
   
   ​    --base_model $MODEL_DIR \
   
   ​    --tokenizer_path baichuan-inc/Baichuan-7B \
   
   ​    --data_file $DATA_FILE \
   
   ​    --with_prompt \
   
   ​    --predictions_file $PREDICTIONS_FILE \
   
   ​    --template_id ${template_id} \
   
   done
   ```

   

3. demo

```
python demo.py
```

