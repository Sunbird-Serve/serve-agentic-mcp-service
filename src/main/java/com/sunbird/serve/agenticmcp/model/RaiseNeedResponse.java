package com.sunbird.serve.agenticmcp.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RaiseNeedResponse {
    private String needId;
    private String message;
    private String status;
    private String error;
} 