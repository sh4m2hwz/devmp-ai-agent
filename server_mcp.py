import speakeasy
from triton import *
from mcp.server.fastmcp import FastMCP, Context
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uuid
import ollama

class AnalyzerVMP:
    def __init__(self,binary_file_path: str):
        self.se = speakeasy.Speakeasy()
        self.module = self.se.load_module(binary_file_path)
        self.se.add_code_hook(self._instrument)
        self.trx = TritonContext(ARCH.X86_64)
        self.trx.setMode(MODE.ALIGNED_MEMORY, True)
        self.is_init_state = True
    def _deobfuscate(self):
        dup_insns = []
        sexprs = self.trx.getSymbolicExpressions()
        keys = sorted(sexprs)
        for key in keys:
            ty = sexprs[key].getType()
            if ty == SYMBOLIC.REGISTER_EXPRESSION or \
            ty == SYMBOLIC.VOLATILE_EXPRESSION or \
            ty == SYMBOLIC.MEMORY_EXPRESSION:
                disasm = sexprs[key].getDisassembly()
                if len(disasm) > 0:
                    dup_insns.append(disasm)
        insns = []
        for i in range(len(dup_insns)):
            if dup_insns[i] not in insns:
                insns.append(dup_insns[i])
        return insns
    def _instrument(self,emu, address, size, user_data):
        if self.is_init_state:
            self.is_init_state = False
            for mem_map in self.se.get_mem_maps():
                content = self.se.mem_read(mem_map.base,mem_map.size)
                self.trx.setConcreteMemoryAreaValue(mem_map.base,content)
            self.trx.setConcreteRegisterValue(self.trx.registers.rax,self.se.reg_read('rax'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rcx,self.se.reg_read('rcx'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rdx,self.se.reg_read('rdx'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rbx,self.se.reg_read('rbx'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rsp,self.se.reg_read('rsp'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rbp,self.se.reg_read('rbp'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rsi,self.se.reg_read('rsi'))
            self.trx.setConcreteRegisterValue(self.trx.registers.rdi,self.se.reg_read('rdi'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r8,self.se.reg_read('r8'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r9,self.se.reg_read('r9'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r10,self.se.reg_read('r10'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r11,self.se.reg_read('r11'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r12,self.se.reg_read('r12'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r13,self.se.reg_read('r13'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r14,self.se.reg_read('r14'))
            self.trx.setConcreteRegisterValue(self.trx.registers.r15,self.se.reg_read('r15'))
        insn = Instruction(address,self.se.mem_read(address,size))
        self.trx.processing(insn)
        opcode = insn.getType()
        ops = insn.getOperands()
        if opcode == OPCODE.X86.RET or \
        opcode == OPCODE.X86.JMP and ops[0].getType() == OPERAND.REG:
            self.se.stop()
    def run_one_vmp_handler(self):
        self.se.run_module(self.module)
        trace = self._deobfuscate()
        self.trx.reset()
        self.is_init_state = True
        return ''.join([insn.split(": ")[1]+"\n" for insn in trace])

@dataclass
class AppContext:
    """Application context for storing active sessions"""
    sessions: Dict[str, AnalyzerVMP]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with shared session state"""
    # Initialize on startup
    sessions = {}
    try:
        yield AppContext(sessions=sessions)
    finally:
        # Cleanup all sessions on shutdown
        for session_id, session in list(sessions.items()):
            await session.cleanup()
            sessions.pop(session_id, None)

mcp = FastMCP("devmp-agent", lifespan=app_lifespan)

# Helper function to get session from context
def get_session(ctx: Context, session_id: str):
    """Get a session by ID or raise an error if not found"""
    # Access the lifespan context through request_context
    sessions = ctx.request_context.lifespan_context.sessions
    if session_id not in sessions:
        raise ValueError(f"No active vmp session with ID: {session_id}")
    return sessions[session_id]

@mcp.tool()
async def create_vmp_session(ctx: Context, path: str):
    session_id = str(uuid.uuid4())
    try:
        session = AnalyzerVMP(path)
        insns = session.run_one_vmp_handler()
    except Exception as e:
        return f"vmp devirt error create session: {e}"
    ctx.request_context.lifespan_context.sessions[session_id] = session
    return f"session started with ID: {session_id}\n\nanalyze vmprotect vmenter handler and get Virtual registers: virtual instruction pointer,virtual stack pointer,ROLL KEY mapped to x86_64 registers: {insns}\n\n"

@mcp.tool()
async def run_one_vmp_handler(ctx: Context,session_id: str):
    try:
        session = get_session(ctx,session_id)
        result = session.run_one_vmp_handler()
    except Exception as e:
        return f"vmp devirt error run one vmp handler: {e}"
    return f"analyze vmprotect handler simplified x86-64 asm code and get pseudo vm instruction:\n\n{result}\n\n"

@mcp.tool()
async def terminate_session(ctx: Context, session_id: str):
    try:
        del ctx.request_context.lifespan_context.sessions[session_id]
    except Exception as e:
        return f"vmp devirt error delete session: {e}"
    return f"complete delete session: {session_id}"

@mcp.tool()
async def analyze_vmp_handler(ctx: Context,session_id: str,code: str):
    try:
        client = ollama.AsyncClient()
        result = await client.generate(model=f'deepseek-r1:32b',prompt='analyze and describe vm bytecode of vmprotect:{code}')
    except Exception as e:
        return f"vmp devirt analyze error {e}"
    return f"description vm bytecode of vmprotect: {result['response']}"

if __name__ == '__main__':
    mcp.run()